# Machine Learning
from sklearn.model_selection import train_test_split
# Deep Learning
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

def processData(train_df, test_df):
    
    # Normalize pixel values to [0, 1] range
    X_train_full = train_df.drop('label', axis=1).values / 255.0
    y_train_full = train_df['label'].values
    X_test = test_df.values / 255.0

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.15, 
        random_state=42, 
        stratify=y_train_full
    )

    # Reshape data for CNN (add channel dimension)
    X_train_cnn = X_train.reshape(-1, 28, 28, 1)
    X_val_cnn = X_val.reshape(-1, 28, 28, 1)
    X_test_cnn = X_test.reshape(-1, 28, 28, 1)

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, 10)
    y_val_cat = to_categorical(y_val, 10)

    # Create data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=10,          # Random rotation ±10 degrees
        width_shift_range=0.1,      # Random horizontal shift
        height_shift_range=0.1,     # Random vertical shift
        zoom_range=0.1,             # Random zoom
        shear_range=0.1            # Random shear
    )

    # Fit the generator on training data
    datagen.fit(X_train_cnn)

    return X_train_cnn, y_train_cat, X_val_cnn, y_val_cat, datagen, X_test_cnn