from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from repositories.postgres.config.db_model import VendorAccount, VendorTag, RequestProcurement, RequestTag
from api.schemas import um_schemas
from datetime import datetime, timezone
from sqlmodel import select, delete
from sqlalchemy.orm import selectinload
from sqlalchemy import func
from sqlalchemy import text
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from utils.auth import create_hashed_password



async def get_vendor_account_list(
    db: AsyncSession, page: int = 1, limit: int = 10
):
    try:
        offset = (page - 1) * limit
        stmt = (
            select(VendorAccount)
            .options(selectinload(VendorAccount.tags))
            .order_by(VendorAccount.id.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(stmt)
        vendors = result.scalars().all()
        return vendors
    except Exception as e:
        raise ValueError(f"failed to fetch vendor list: {str(e)}")


async def get_vendor_account_detail(email: str, db: AsyncSession):
    try:
        stmt = (
            select(VendorAccount)
            .options(selectinload(VendorAccount.tags))
            .where(VendorAccount.email == email)
        )
        result = await db.execute(stmt)
        vendor = result.scalars().first()
        return vendor
    except Exception as e:
        raise ValueError(f"failed to fetch vendor detail: {str(e)}")


async def create_vendor_account(log, payload: um_schemas.VendorRegister, db: AsyncSession) -> VendorAccount:
    now = datetime.now()

    try:
        # pre-check for duplicates
        stmt = select(VendorAccount).where(VendorAccount.email == payload.email)
        result = await db.execute(stmt)
        if result.scalars().first():
            raise ValueError("email already registered")

        # create vendor
        vendor = VendorAccount(
            name=payload.name,
            description=payload.description,
            location=payload.location,
            email=payload.email,
            hashed_password=create_hashed_password(payload.password),
            created_at=now,
            updated_at=now
        )
        db.add(vendor)

        # optional tags
        if payload.tags:
            tag_objs = [
                VendorTag(tag_name=tag.strip(), vendor_email=payload.email)
                for tag in payload.tags
            ]
            db.add_all(tag_objs)

        await db.commit()
        await db.refresh(vendor)
        return vendor

    except IntegrityError as e:
        await db.rollback()
        log.error(f"IntegrityError: {str(e)}")
        raise ValueError("duplicate entry detected, please retry")
    except Exception as e:
        await db.rollback()
        log.error(str(e), exc_info=True)
        raise ValueError(str(e))

async def create_procurement_request(log, payload: um_schemas.RequestProcurementCreate, db: AsyncSession) -> int:
    try:
        due_date_naive = payload.due_date.astimezone(timezone.utc).replace(tzinfo=None)
        new_request = RequestProcurement(
            title=payload.title,
            category=payload.category,
            description=payload.description,
            requirements=payload.requirements,
            price=payload.price,
            location=payload.location,
            due_date=due_date_naive,
            status="open",
            created_at=datetime.now(),
            updated_at=None,
        )

        db.add(new_request)
        await db.flush()  # to get new_request.id

        tags = [
            RequestTag(request_id=new_request.id, tag_name=tag.strip().lower())
            for tag in payload.tags
        ]
        db.add_all(tags)

        await db.commit()
        return new_request.id

    except IntegrityError as e:
        await db.rollback()
        log.error(f"IntegrityError: {str(e)}")
        raise ValueError("duplicate entry detected, please retry")
    except Exception as e:
        await db.rollback()
        log.error(str(e), exc_info=True)
        raise ValueError(str(e))

async def update_vendor_account(log, email: str, payload: um_schemas.VendorRegister, db: AsyncSession) -> VendorAccount:
    now = datetime.now()

    try:
        # fetch existing vendor by email
        stmt = select(VendorAccount).where(VendorAccount.email == email)
        result = await db.execute(stmt)
        vendor = result.scalars().first()

        if not vendor:
            raise ValueError("vendor not found")

        # update fields
        vendor.name = payload.name
        vendor.description = payload.description
        vendor.location = payload.location
        vendor.updated_at = now

        # update tags if present
        if payload.tags is not None:
            # delete existing tags for this vendor
            await db.execute(delete(VendorTag).where(VendorTag.vendor_email == vendor.email))
            # add new tags
            tag_objs = [
                VendorTag(tag_name=tag.strip(), vendor_email=vendor.email)
                for tag in payload.tags
            ]
            db.add_all(tag_objs)

        await db.commit()
        await db.refresh(vendor)
        return vendor

    except IntegrityError as e:
        await db.rollback()
        log.error(f"IntegrityError: {str(e)}")
        raise ValueError("duplicate entry detected, please retry")
    except Exception as e:
        await db.rollback()
        log.error(str(e), exc_info=True)
        raise ValueError(str(e))

async def get_procurement_request_list(
    db: AsyncSession, page: int = 1, limit: int = 10
):
    try:
        offset = (page - 1) * limit

        # query for data
        stmt = (
            select(RequestProcurement)
            .options(selectinload(RequestProcurement.tags))
            .order_by(RequestProcurement.id.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(stmt)
        requests = result.scalars().all()

        # query for total count
        count_stmt = select(func.count()).select_from(RequestProcurement)
        total_result = await db.execute(count_stmt)
        total_count = total_result.scalar()

        return um_schemas.ProcListItems(
            total_count=total_count,
            vendors=requests
        )
    except Exception as e:

        raise ValueError(f"failed to fetch procurement request list: {str(e)}")


from nbeats_pytorch.model import NBeatsNet
import torch
from dateutil.relativedelta import relativedelta
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
backcast_length = 5 
forecast_length = 3 
model = NBeatsNet(
    device=device,
    stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
    forecast_length=forecast_length,
    backcast_length=backcast_length,
    hidden_layer_units=128
).to(device)

async def get_procurement_request_list_forecast(
    log,
    db: AsyncSession
):
    try:
        stmt = text("""
            SELECT 
                date_trunc('month', due_date) AS month,
                SUM(price) AS total_price
            FROM requestprocurement
            GROUP BY month
            ORDER BY month
        """)

        result = await db.execute(stmt)
        rows = result.all()

        monthly_sums = [
            {
                "month": row.month.date().isoformat(),
                "total_price": float(row.total_price),
            }
            for row in rows
        ]
        # convert to dataframe
        df_old = pd.DataFrame(monthly_sums)
        df_old['month'] = pd.to_datetime(df_old['month'])
        y_new = df_old['total_price'].to_numpy()
        x_input = torch.tensor(y_new[-backcast_length:], dtype=torch.float32).unsqueeze(0).to(device)

        model.load_state_dict(torch.load('./model/nbeats_model_pengadaan.pt', map_location=device))
        model.eval()
        # inference
        with torch.no_grad():
            _, forecast = model(x_input)

        forecast_np = forecast.cpu().numpy().flatten()
        print("Forecast:", forecast_np)
        print(df_old)
        df_old['type'] = 'old'
        df_old['month'] = pd.to_datetime(df_old['month'])
        # Prepare forecast entries
        last_date = df_old['month'].max()
        forecast_months = [last_date + relativedelta(months=i+1) for i in range(len(forecast_np))]

        df_forecast = pd.DataFrame({
            'month': forecast_months,
            'total_price': forecast_np,
            'type': 'forecasted'
        })

        # Combine
        df_all = pd.concat([df_old, df_forecast], ignore_index=True)
        return df_all.to_dict(orient='records')


    except Exception as e:
        log.error(str(e), exc_info=True)
        raise ValueError(f"failed to fetch procurement request list: {str(e)}")



from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import re

# Setup Sastrawi stemmer and stopword remover
stemmer = StemmerFactory().create_stemmer()
indo_stopwords = set(StopWordRemoverFactory().get_stop_words())
eng_stopwords = set(ENGLISH_STOP_WORDS)

# Combine stopwords
combined_stopwords = indo_stopwords.union(eng_stopwords)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    filtered = [word for word in words if word not in combined_stopwords]
    filtered_text = " ".join(filtered)
    stemmed_text = stemmer.stem(filtered_text)
    return stemmed_text

def tag_overlap_score(account_tags, tender_tags):
    common = set(account_tags) & set(tender_tags)
    total = set(account_tags) | set(tender_tags)
    return len(common) / len(total) if total else 0


def combined_similarity(account, tenders):
    # Preprocess descriptions
    corpus = [preprocess_text(account["description"])] + [
        preprocess_text(t["description"]) for t in tenders
    ]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    account_vec = tfidf_matrix[0]
    tender_vecs = tfidf_matrix[1:]

    results = []
    for i, tender in enumerate(tenders):
        desc_sim = cosine_similarity(account_vec, tender_vecs[i])[0][0]
        tag_sim = tag_overlap_score(account["tags"], tender["tags"])
        combined = 0.6 * desc_sim + 0.4 * tag_sim
        tender_with_score = tender.copy()
        tender_with_score["score"] = combined
        results.append(tender_with_score)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


async def get_procurement_request_list_smart(
    log, email, db: AsyncSession, page: int = 1, limit: int = 10
):
    try:
        # get account detail 
        stmt = (
            select(VendorAccount)
            .options(selectinload(VendorAccount.tags))
            .where(VendorAccount.email == email)
        )
        result = await db.execute(stmt)
        account = result.scalars().first()
        account = account.to_dict() if account else None

        # query for data
        stmt = (
            select(RequestProcurement)
            .options(selectinload(RequestProcurement.tags))
        )
        result = await db.execute(stmt)
        requests = result.scalars().all()
        tenders = [r.to_dict() for r in requests]

  
        tenders_with_rank = combined_similarity(account, tenders)

        offset = (page - 1) * limit
        paginated_tenders = tenders_with_rank[offset : offset + limit]

        total_count = len(tenders)

        return {
            'total_count': total_count,
            'vendors': paginated_tenders
        }
    except Exception as e:
        log.error(str(e), exc_info=True)
        raise ValueError(f"failed to fetch procurement request list: {str(e)}")
