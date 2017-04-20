from sklearn.preprocessing import LabelEncoder


class ExtendedLabelEncoder(LabelEncoder):

    def __init__(self):
        super(ExtendedLabelEncoder, self).__init__()
        self.encoded_classes_ = None

    def fit(self, y):
        ret = super().fit(y)
        self.encoded_classes_ = self.transform(self.classes_)
        return ret

    def fit_transform(self, y):
        ret = super().fit_transform(y)
        self.encoded_classes_ = self.transform(self.classes_)
        return ret
