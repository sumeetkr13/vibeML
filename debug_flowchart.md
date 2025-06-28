# Application Flow Analysis

## Current Flow When "Train Model" Button is Clicked:

```
1. User clicks "🚀 Train Model" button (app.py line 383)
   ↓
2. st.spinner starts: "Training model... This may take a few minutes."
   ↓
3. try block begins (app.py line 385)
   ↓
4. Debug info displayed:
   - DataFrame type: <class 'pandas.core.frame.DataFrame'>
   - Features type: <class 'list'>
   - Target column: Personality
   - DataFrame shape: (2900, 8)
   ↓
5. train_model() called (app.py line 393-395)
   ↓
6. train_model execution:
   - Debug shows DataFrame created successfully
   - Train/test split works
   - Pipeline fits successfully
   - Returns without error
   ↓
7. Results unpacked (app.py line 396)
   ↓
8. ERROR OCCURS HERE: "Specifying the columns using strings is only supported for dataframes"
   ↓
9. Exception caught and displayed in UI
```

## The Mystery:

The error message appears in Streamlit UI but NOT in the console debug logs. This suggests:

1. **Error occurs AFTER train_model returns successfully**
2. **Error happens during result unpacking or immediate next step**
3. **The error might be in a different thread or async operation**

## Possible Error Sources:

1. **Line 396**: `model, X_train, X_test, y_train, y_test, preprocessor, label_encoder = results`
2. **Line 399**: Prediction step
3. **Session state storage** (lines 402-409)
4. **evaluate_model call** (line 414)
5. **Streamlit internal operations**

## Investigation Needed:

Need to add try-catch around EACH individual line after train_model to isolate the exact error location.