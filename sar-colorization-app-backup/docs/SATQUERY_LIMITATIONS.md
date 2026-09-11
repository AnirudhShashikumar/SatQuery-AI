# SatQuery Scientific Limitations

## General

- Outputs are decision-support artifacts, not ground truth or operational ISRO products.
- GeoVision does not infer hidden locations from pixels or filenames.
- Uploaded filenames are never used as semantic evidence.
- Confidence scores are omitted when no calibrated score exists.
- Reduced analysis grids can omit small features.

## Captioning

- The RSICD-fine-tuned BLIP caption is model-generated and can omit or misidentify objects.
- It supports optical and RGB-like multispectral representations, not general SAR captioning.
- Its checkpoint and license provenance are disclosed with each result.

## Controlled VQA

- Water support can include cloud shadow, terrain shadow, dark roofs, or other dark smooth surfaces.
- Structural support can include bright soil, roads, speckle, or vegetation structure; it does not confirm buildings.
- Visible green support is not NDVI and does not prove vegetation without suitable spectral bands.
- Agricultural support is a field-like texture heuristic and cannot identify crop type or confirm land use.
- Controlled VQA is not RSVQA-fine-tuned and never reports high calibrated semantic confidence.

## Text-guided grounding

- Grounding DINO is used zero-shot and is not fine-tuned or calibrated for remote-sensing imagery.
- Small, dense, low-contrast, or unusual-scale objects can be missed or localized imprecisely.
- Alignment scores are not calibrated scientific probabilities, and predicted boxes are not ground truth.
- Optical/RGB-like multispectral imagery is supported; SAR grounding and SAM/SAM2 mask refinement are not connected.

## Bi-temporal change

- The engine measures normalized visual/radiometric difference, not semantic land-cover transition.
- It does not identify flood, construction, deforestation, damage, or causal mechanism.
- Images requiring registration, reprojection, or resampling are not silently altered.
- Atmospheric, seasonal, illumination, sensor, or processing differences can appear as change.

## Optical–SAR fusion

- SAR values are relative uncalibrated intensities unless the source data itself provides calibrated values; GeoVision does not claim physical backscatter.
- Low relative intensity can include smooth surfaces, radar shadow, or terrain effects.
- High intensity/heterogeneity can include vegetation structure, speckle, or isolated scatterers.
- Pixel-level fusion requires exact grids and compatible geospatial metadata.

## Reports and caching

- Reports embed prepared display/evidence products rather than original uploads.
- JSON preview URLs are temporary; the ZIP package embeds the corresponding evidence files.
- Results and artifacts expire from bounded local storage and are lost on backend restart.
- A cached response is valid only for its complete content-derived key and is clearly labelled.

## Explicitly unavailable

- SAR text-guided grounding and grounding mask refinement.
- General single-image SAR VQA.
- General SAR captioning.
- Calibrated semantic segmentation.
- Unsupported causal change interpretation.
- Public benchmark or operational deployment claims not backed by measured evidence.
