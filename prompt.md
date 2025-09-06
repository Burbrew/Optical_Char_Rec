You are given images containing text in various formats, fonts, orientations, and quality levels. Your task is to extract all visible text from each image accurately.

## Task Requirements

For each input image, you must:
1. Extract all visible text content
2. Provide a confidence score (0.0 to 1.0) for the extraction quality
3. Handle various text orientations, fonts, and image qualities
4. Output results in the specified JSON format

## Output Format

Create a JSON file named `solution.json` with the following structure:

```json
{
  "predictions": {
    "<filename>": {
      "text": "<extracted_text>",
      "confidence": <float_0_to_1>
    },
    ...
  }
}
```

## Image Types and Challenges

The test images will include:
- **Clean printed text** - High quality, clear fonts
- **Handwritten text** - Various handwriting styles
- **Rotated text** - Text at different angles
- **Low resolution text** - Blurry or pixelated images
- **Mixed fonts** - Different font sizes and styles
- **Background noise** - Text on textured or colored backgrounds
- **Multiple languages** - English and other languages
- **Document text** - Formatted documents with tables/lists

## Constraints and Tips

- Extract text as accurately as possible, preserving original formatting when relevant
- Provide realistic confidence scores based on image quality and text clarity
- Handle empty images by returning empty text with low confidence
- For rotated text, try to detect and correct orientation
- Use appropriate OCR tools (Tesseract, EasyOCR, etc.) based on image characteristics
- Consider preprocessing images (denoising, contrast enhancement) if needed

## Evaluation Criteria

Your solution will be evaluated on:
- Text extraction accuracy (character and word level)
- Confidence score calibration
- Handling of various image challenges
- Proper JSON formatting

## Example

For an image containing "Hello World" with high clarity:
```json
{
  "predictions": {
    "image_1.png": {
      "text": "Hello World",
      "confidence": 0.95
    }
  }
}
```
