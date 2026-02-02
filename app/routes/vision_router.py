from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from app.services.image_analysis_service import analyze_medical_image

router = APIRouter(prefix="/api/v1/vision", tags=["Vision / Multimodal"])

@router.post("/analyze", summary="Analizar imagen médica")
async def analyze_image_endpoint(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form("Describe los hallazgos médicos en esta imagen de manera detallada.")
):
    """
    Sube una imagen (PNG, JPG, etc.) y obtén una descripción clínica.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    content = await file.read()
    
    result_text = await analyze_medical_image(content, prompt=prompt)
    
    return {
        "filename": file.filename,
        "analysis": result_text
    }
