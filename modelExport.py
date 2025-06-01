import tensorflow as tf
import os
import shutil

# List of models with name, class, and required input shape
models = [
    ("MobileNetV2", tf.keras.applications.MobileNetV2, (224, 224, 3)),
    ("ResNet50", tf.keras.applications.ResNet50, (224, 224, 3)),
    ("EfficientNetB0", tf.keras.applications.EfficientNetB0, (224, 224, 3)),
    ("DenseNet121", tf.keras.applications.DenseNet121, (224, 224, 3)),
    ("InceptionV3", tf.keras.applications.InceptionV3, (299, 299, 3)),
    ("Xception", tf.keras.applications.Xception, (299, 299, 3)),
    ("NASNetMobile", tf.keras.applications.NASNetMobile, (224, 224, 3)),
]

base_dir = "exported_models"
os.makedirs(base_dir, exist_ok=True)

for model_name, model_class, input_shape in models:
    try:
        print(f"\n🔧 Exporting: {model_name}")
        model = model_class(weights="imagenet", input_shape=input_shape)

        model_dir = os.path.join(base_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Save as .keras
        keras_path = os.path.join(model_dir, f"{model_name}.keras")
        model.save(keras_path)
        print(f"✅ Saved .keras: {keras_path}")

        # Save as .h5
        h5_path = os.path.join(model_dir, f"{model_name}.h5")
        model.save(h5_path)
        print(f"✅ Saved .h5: {h5_path}")

        # Save as SavedModel (using export for Keras 3.x)
        savedmodel_dir = os.path.join(model_dir, "saved_model")
        model.export(savedmodel_dir)
        print(f"✅ Saved SavedModel: {savedmodel_dir}")

        # Zip the SavedModel directory
        zip_path = shutil.make_archive(os.path.join(model_dir, "saved_model"), 'zip', savedmodel_dir)
        print(f"✅ Zipped SavedModel: {zip_path}")

    except Exception as e:
        print(f"❌ Skipped {model_name}: {e}")