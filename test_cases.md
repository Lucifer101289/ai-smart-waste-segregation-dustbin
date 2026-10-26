# Test Cases

These are manual test cases to run after training the model.

| ID | Input | Expected behaviour |
|---|---|---|
| TC-01 | Clear image of food scraps, labelled organic in training | Predicts organic if model learned the class |
| TC-02 | Clear image of a clean plastic bottle, labelled recyclable | Predicts recyclable if model learned the class |
| TC-03 | Unsupported item, such as an electronic device | May be wrong; document the result and do not assume either category is safe |
| TC-04 | Blurry or dark image | App returns a prediction; low confidence should trigger a manual check |
| TC-05 | No model file present | App shows a helpful warning and stops |
| TC-06 | Unsupported file type | File uploader should not accept it |
| TC-07 | Held-out test images | Record accuracy and errors; do not use training images for this test |

## Evaluation checklist
- Keep a separate test set that is not used during training.
- Report accuracy and per-class precision/recall.
- Inspect false predictions, especially organic waste classified as recyclable and vice versa.
- Do not claim accuracy results until the model has actually been trained and evaluated.
