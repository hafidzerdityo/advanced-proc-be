from fastapi import APIRouter, Depends, HTTPException, Body, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import List
from utils.auth import verify_password, create_access_token, get_current_user
from datetime import timedelta

from api.schemas import um_schemas
from services.user_management import (
    create_vendor_account,
    update_vendor_account,
    get_vendor_account_detail,
    get_vendor_account_list,
    create_procurement_request,
    get_procurement_request_list,
    get_procurement_request_list_smart,
    get_procurement_request_list_forecast
)
from api.deps import get_session
from logs.log_config import get_logger

router = APIRouter()
log = get_logger()



@router.get("/vendor/check_token", tags=["Vendor"])
async def check_vendor(current_user: dict = Depends(get_current_user)):
    if "error" in current_user and current_user["error"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=current_user.get("msg", "invalid authentication credentials"),
            headers={"WWW-Authenticate": "Bearer"},
        )

    email = current_user.get("sub")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing user info in token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "resp_msg": "token valid",
        "resp_data": {"email": email}
    }


@router.post("/vendor/login", tags=["Vendor"])
async def vendor_login(
    payload: um_schemas.VendorLogin = Body(...),
    db: AsyncSession = Depends(get_session),
):
    try:
        vendor = await get_vendor_account_detail(payload.email, db)
        if not vendor:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"resp_msg": "Email atau Password Salah", "resp_data": None},
            )

        if not verify_password(payload.password, vendor.hashed_password):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"resp_msg": "Email atau Password Salah", "resp_data": None},
            )

        access_token_expires = timedelta(minutes=10000)  # or from your config
        token = create_access_token(data={"sub": vendor.email}, expires_delta=access_token_expires)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "resp_msg": "Login Berhasil",
                "resp_data": {
                    "access_token": token,
                    "token_type": "bearer",
                },
            },
        )

    except Exception as e:
        log.error(str(e))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"resp_msg": str(e), "resp_data": None},
        )

@router.post("/vendor/user", tags=['Vendor'], status_code=status.HTTP_201_CREATED)
async def register_vendor(
    payload: um_schemas.VendorRegister = Body(...),
    db: AsyncSession = Depends(get_session),
):
    try:
        await create_vendor_account(log, payload, db)
        return {
            'resp_msg': "registration success",
            'resp_data': None
        }
    except Exception as e:
        log.error(str(e))
        return JSONResponse(
            status_code=400,
            content={"resp_msg": str(e), "resp_data": None},
        )

@router.post("/admin/request/procurement", tags=['Admin'])
async def register_procurement(
    payload: um_schemas.RequestProcurementCreate = Body(...),
    db: AsyncSession = Depends(get_session),
):
    try:
        await create_procurement_request(log, payload, db)
        return {
            'resp_msg': "registration success",
            'resp_data': None
        }
    except Exception as e:
        log.error(str(e))
        return JSONResponse(
            status_code=400,
            content={"resp_msg": str(e), "resp_data": None},
        )


@router.put("/vendor/user", tags=["Vendor"])
async def update_vendor(
    payload: um_schemas.VendorUpdate = Body(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    if "error" in current_user and current_user["error"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=current_user.get("msg", "invalid authentication credentials"),
            headers={"WWW-Authenticate": "Bearer"},
        )

    email = current_user.get("sub")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing user info in token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        vendor = await update_vendor_account(log, email, payload, db)
        return {
            "resp_msg": "update success",
            "resp_data": vendor
        }
    except Exception as e:
        log.error(str(e))
        return JSONResponse(
            status_code=400,
            content={"resp_msg": str(e), "resp_data": None},
        )


@router.get("/vendor/user/detail", response_model=um_schemas.VendorDetailResponse, tags=["Vendor"])
async def vendor_detail(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    if "error" in current_user and current_user["error"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=current_user.get("msg", "invalid authentication credentials"),
            headers={"WWW-Authenticate": "Bearer"},
        )
    email = current_user.get("sub")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing user info in token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        vendor = await get_vendor_account_detail(email, db)
        if not vendor:
            raise HTTPException(status_code=404, detail="vendor not found")
        return {"resp_msg": "ok", "resp_data": vendor}
    except Exception as e:
        log.error(str(e))
        return JSONResponse(
            status_code=400,
            content={"resp_msg": str(e), "resp_data": None},
        )

@router.post("/vendor/user/list", response_model=um_schemas.VendorListResponse, tags=["Vendor"])
async def vendor_list(
    payload: um_schemas.VendorListRequest = Body(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    try:
        if "error" in current_user and current_user["error"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=current_user.get("msg", "invalid authentication credentials"),
                headers={"WWW-Authenticate": "Bearer"},
            )
        email = current_user.get("sub")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing user info in token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        vendors = await get_vendor_account_list(
            db, page=payload.page, limit=payload.limit
        )
        return {"resp_msg": "ok", "resp_data": vendors}
    except Exception as e:
        log.error(str(e))
        return JSONResponse(
            status_code=400,
            content={"resp_msg": str(e), "resp_data": None},
        )

@router.get("/admin/request/list_forecast", tags=["Admin"])
async def procurement_request_list(
    db: AsyncSession = Depends(get_session),
):
    try:
        requests = await get_procurement_request_list_forecast(log, db)
        return {"resp_msg": "ok", "resp_data": requests}
    except Exception as e:
        log.error(str(e))
        return JSONResponse(
            status_code=400,
            content={"resp_msg": str(e), "resp_data": None},
        )

# @router.post("/vendor/request/list", response_model=um_schemas.RequestProcurementListResponse, tags=["Vendor"])
# async def procurement_request_list(
#     payload: um_schemas.RequestProcurementListRequest = Body(...),
#     current_user: dict = Depends(get_current_user),
#     db: AsyncSession = Depends(get_session),
# ):
#     try:
#         if "error" in current_user and current_user["error"]:
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail=current_user.get("msg", "invalid authentication credentials"),
#                 headers={"WWW-Authenticate": "Bearer"},
#             )
#         email = current_user.get("sub")
#         if not email:
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="missing user info in token",
#                 headers={"WWW-Authenticate": "Bearer"},
#             )

#         requests = await get_procurement_request_list(
#             db, page=payload.page, limit=payload.limit
#         )
#         return {"resp_msg": "ok", "resp_data": requests}
#     except Exception as e:
#         log.error(str(e))
#         return JSONResponse(
#             status_code=400,
#             content={"resp_msg": str(e), "resp_data": None},
#         )

@router.post("/vendor/request/smart_list", tags=["Vendor"])
async def procurement_request_list(
    payload: um_schemas.RequestProcurementListRequest = Body(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    try:
        if "error" in current_user and current_user["error"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=current_user.get("msg", "invalid authentication credentials"),
                headers={"WWW-Authenticate": "Bearer"},
            )
        email = current_user.get("sub")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing user info in token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        requests = await get_procurement_request_list_smart(
            log, email, db, payload.page, payload.limit
        )
        return {"resp_msg": "ok", "resp_data": requests}
    except Exception as e:
        log.error(str(e))
        return JSONResponse(
            status_code=400,
            content={"resp_msg": str(e), "resp_data": None},
        )
