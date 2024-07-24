from data.prepare_data import prepare_data
from model.transfer_learning_model import create_model

def train_model():
    train_generator, validation_generator = prepare_data()
    model = create_model()
    model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )
    model.save('transfer_learning_model.h5')

if __name__ == '__main__':
    train_model()
