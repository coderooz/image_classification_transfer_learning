import tensorflow as tf
from data.prepare_data import prepare_data

def evaluate_model():
    _, validation_generator = prepare_data()
    model = tf.keras.models.load_model('transfer_learning_model.h5')
    test_loss, test_acc = model.evaluate(validation_generator)
    print(f'Test accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    evaluate_model()
