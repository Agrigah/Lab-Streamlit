"""Train the architecture supplied in the fruit CNN notebook and export it."""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=BASE_DIR / 'fruits')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    tf.keras.utils.set_random_seed(42)
    datasets = {}
    classes = None
    for split in ('train', 'validation', 'test'):
        dataset = tf.keras.utils.image_dataset_from_directory(
            args.data_dir / split, image_size=(32, 32), batch_size=20,
            shuffle=(split == 'train'), seed=42,
        )
        if classes is None:
            classes = dataset.class_names
        elif dataset.class_names != classes:
            raise ValueError(f'Class folders differ in {split}')
        datasets[split] = dataset.prefetch(tf.data.AUTOTUNE)
    if len(classes) != 3:
        raise ValueError(f'Expected three fruit classes, found {classes}')
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(32, 32, 3)),
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, (3, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, (3, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, (3, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    history = model.fit(datasets['train'], validation_data=datasets['validation'], epochs=args.epochs, verbose=2)
    metrics = model.evaluate(datasets['test'], return_dict=True, verbose=2)
    model.save(BASE_DIR / 'CNN_fruits.keras')
    model.save(BASE_DIR / 'CNN_fruits.h5')
    (BASE_DIR / 'class_names.json').write_text(json.dumps(classes, indent=2), encoding='utf-8')
    (BASE_DIR / 'training_results.json').write_text(json.dumps({
        'seed': 42, 'epochs': args.epochs, 'class_names': classes,
        'test_metrics': metrics, 'history': history.history,
    }, indent=2), encoding='utf-8')
    sample, _ = next(iter(datasets['test']))
    restored = tf.keras.models.load_model(BASE_DIR / 'CNN_fruits.keras', compile=False)
    np.testing.assert_allclose(model(sample, training=False).numpy(), restored(sample, training=False).numpy(), rtol=1e-5, atol=1e-6)
    print('Saved and verified CNN_fruits.keras and CNN_fruits.h5. Class order:', classes)


if __name__ == '__main__':
    main()
