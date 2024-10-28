# %%
import tensorflow as tf
from tensorflow.keras import layers,models,optimizers,losses
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image_dataset_from_directory,image
import numpy as np
from Eval_metrics_gen_excel import save_predictions_to_excel,generate_metrics_report
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import platform

# %%
try:
    import google.colab
    on_colab = True
except ImportError:
    on_colab = False

if on_colab:
    base_dir = '/content/drive/MyDrive/Colab Notebooks/'
    train_dir=  base_dir+'Dataset/training'
    val_dir=    base_dir+'Dataset/validation'
    save_dir=   base_dir+'Resources/'
    model_path= save_dir+'efc_model.keras'
    print("Running on Google Colab. Training directory set to:", train_dir)
    # Import Colab-specific libraries if needed
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

else:
    base_dir = ''
    train_dir=  base_dir+'Dataset/training'
    val_dir=    base_dir+'Dataset/validation'
    save_dir=   base_dir+'Dataset/'
    model_path= save_dir+'efc_model.keras'
    print("Running on Local Host. Training directory set to:", train_dir)

# %%
#model_path= r'C:\Media files\Coding\ML\BIOML\Capsule-Vision-2024-Challenge\Dataset\efc_model.keras'

# %%
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = True
model = models.Sequential()

model.add(base_model)
model.add(layers.Reshape((49, 1280)))  # Ensure it matches your LSTM input requirement

model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.LSTM(32))

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

input_tensor = tf.keras.Input(shape=(224, 224, 3)) 
output_tensor = model(input_tensor)

model.summary()


# %%
model.load_weights(model_path)

# %%
#modify according to the model type being used
#these data loading functions are specific to VGG16, modify accordingly 
def load_and_preprocess_image(full_path, target_size):
    img = load_img(full_path, target_size=target_size)
    img_array = img_to_array(img)
    # preprocessed_img = preprocess_input(img_array)
    return img_array
def get_data(excel_path=r'C:\Media files\Coding\ML\BIOML\Capsule-Vision-2024-Challenge\Dataset\validation\validation_data.xlsx'):
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['image_path'])
    # if windows replace forward slash with back slash
    if platform.system() == 'Windows':
        df['image_path'] = df['image_path'].apply(lambda x: x.replace('/', os.sep))
    else:
        df['image_path'] = df['image_path'].apply(lambda x: x.replace('\\', os.sep))
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    # X = np.array([load_and_preprocess_image(os.path.join(base_dir, path), image_size) for path in df['image_path'].values])
    # y = df[class_columns].values
    return df
def load_test_data(test_dir, image_size=(224, 224)):
    image_paths = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.lower().endswith(('jpg'))]
    X_test = np.array([load_and_preprocess_image(path, image_size) for path in image_paths])
    return X_test, image_paths

# %%
img_height = 224
img_width = 224

batch_size = 32

# %%
val_ds = image_dataset_from_directory(
    val_dir,
    label_mode='categorical',  # Use 'categorical' if using one-hot encoded labels
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=False  # Do not shuffle validation data
)


# %%
y_true = np.concatenate([y for x, y in val_ds], axis=0)

# %%
image_size=(224,224)
val_df = get_data(excel_path=r'C:\Media files\Coding\ML\BIOML\Capsule-Vision-2024-Challenge\Dataset\validation\validation_data.xlsx')

# %%
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# %%
df=generate_metrics_report(y_true,y_pred_probs)
print(df)
output_val_predictions="validation_excel.xlsx"
save_predictions_to_excel(val_df['image_path'].values, y_pred_probs, output_val_predictions)

# %%
# For Test data - uncomment when you have test data
test_path = base_dir+'Dataset/Testing set/Images'
image_paths = [os.path.join(test_path, fname) for fname in os.listdir(test_path) if fname.lower().endswith(('jpg'))]
len(image_paths)

# %%
test_ds = image_dataset_from_directory(
    test_path,
    batch_size=batch_size,
    label_mode=None,
    image_size=(img_height, img_width),
    shuffle=False
)

# %%
#X_test, image_paths = load_test_data(test_path,image_size=image_size)
y_test_pred = model.predict(test_ds)

output_test_predictions="test_excel.xlsx"
save_predictions_to_excel(image_paths, y_test_pred, output_test_predictions)