from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class ExtendedLabelEncoder(LabelEncoder):

    def __init__(self):
        super(ExtendedLabelEncoder, self).__init__()
        self.encoded_classes_ = None
        self.classes_ = None

    def fit(self, y):
        if (self.classes_ is not None
                and self.classes_.flags.writeable is False):
            raise ValueError("Encoder has already been fitted.")

        ret = super(ExtendedLabelEncoder, self).fit(y)
        self.encoded_classes_ = self.transform(self.classes_)
        self.classes_.flags.writeable = False
        return ret

    def fit_transform(self, y):
        if (self.classes_ is not None
                and self.classes_.flags.writeable is False):
            raise ValueError("Encoder has already been fitted.")
        ret = super(ExtendedLabelEncoder, self).fit_transform(y)
        self.encoded_classes_ = self.transform(self.classes_)
        self.classes_.flags.writeable = False
        return ret

    def single_inv_transform(self, i):
        check_is_fitted(self, 'classes_')
        if i > self.classes_.size:
            raise ValueError("Index out of bounds for this encoder.")
        return self.classes_[i]
