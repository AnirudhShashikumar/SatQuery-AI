# Cartosat-2S and RISAT Readiness

Status: engineering hardening complete; actual Cartosat-2S/RISAT evaluation evidence is unavailable.

## What the current pipeline supports

- Byte-validated PNG, JPEG, TIFF and GeoTIFF ingestion with bounded upload size.
- uint8, uint16 and floating-point TIFF statistics, finite/NoData masking, CRS, affine transform, bounds, resolution, orientation and overlap inspection.
- Explicit RGB/color-interpretation selection and recognized Sentinel-2-like B4/B3/B2 mapping.
- Conservative unknown-multispectral behavior: inspection-only first-band grayscale instead of silent first-three-band RGB.
- Native single/dual-channel SAR normalization based on finite per-channel scene percentiles without a fixed sensor range.
- Explicit SAR value-domain provenance (`integer_unknown_scale`, `db_like_unverified`, `amplitude_or_power_like_unverified`, or unknown).
- ChangerEx optical bi-temporal inference on MPS/CPU with deterministic fallback.

## Known assumptions and domain-shift risks

- ChangerEx was trained on LEVIR-CD and primarily represents building change; Cartosat scale, radiometry, season and acquisition geometry may differ.
- Grounding, RSVQA, captioning and SVE accept RGB or documented RGB-like visual mappings. Panchromatic or unknown multispectral products require a defensible mapping before semantic use.
- Native SAR evidence is based on relative intensity and texture, not calibrated sigma0/gamma0 backscatter. Incidence angle, speckle, orbit, terrain and polarization can change the response.
- SARFusionFormer requires exactly two explicitly ordered VV/VH channels. Missing channels are not duplicated. Other RISAT polarizations require a compatible model or remain in native-SAR analysis only.
- Equal dimensions do not establish co-registration. CRS, transform, bounds, resolution and orientation are checked independently.
- High-resolution object scale may differ materially from VRSBench and RSVQA imagery.

## Sensor-agnostic preprocessing

- finite/NoData masks;
- scene-relative percentile normalization with recorded limits;
- explicit dtype and band/polarization provenance;
- bounded preview generation separate from preserved source values;
- no silent reprojection, resampling, modality coercion, RGB guessing or SAR channel duplication.

## What remains unverified

- No approved Cartosat-2S or RISAT samples are present in the repository.
- Sensor-specific band metadata naming, calibration units, polarization ordering and geolocation quality have not been validated.
- Model quality, confidence calibration and failure rates under Cartosat/RISAT domain shift are unknown.

Therefore the correct status is **EVIDENCE GAP**. Do not state “RISAT verified” or “Cartosat-2S verified” until an approved representative set is run with reference labels and a documented protocol.
