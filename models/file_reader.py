
class FileReader:
    def __init__(self):
        self.x_train, self.y_train = [], []
        self.x_valid, self.y_valid = [], []
        self.x_test, self.y_test = [], []

    def read_from_file(self):
        train_file = open('./data/train.txt', 'r')
        valid_file = open('./data/valid.txt', 'r')
        test_file = open('./data/test.txt', 'r')

        for line in train_file.readlines():
            words = line.strip().split('\t')
            if len(words) == 2:
                self.x_train.append(words[0])
                self.y_train.append(words[1])

        for line in valid_file.readlines():
            words = line.strip().split('\t')
            if len(words) == 2:
                self.x_valid.append(words[0])
                self.y_valid.append(words[1])

        for line in test_file.readlines():
            words = line.strip().split('\t')
            if len(words) == 2:
                self.x_test.append(words[0])
                self.y_test.append(words[1])


if __name__ == '__main__':
    file_reader = FileReader()
    file_reader.read_from_file()

    print("The training set file: \nInputs: {}\nOutputs: {}\n".format(
        file_reader.x_train[:5], file_reader.y_train[:5]))
    print("The validation set file: \nInputs: {}\nOutputs: {}\n".format(
        file_reader.x_valid[:5], file_reader.y_valid[:5]))
    print("The test set file: \nInputs: {}\nOutputs: {}\n".format(
        file_reader.x_test[:5], file_reader.y_test[:5]))
