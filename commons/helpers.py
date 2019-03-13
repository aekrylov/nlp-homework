from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


class FeatureNamePipeline(Pipeline):

    def get_feature_names(self):
        return self.steps[0][1].get_feature_names()


class NamedTfidfTransformer(TfidfTransformer):

    def get_feature_names(self):
        return self.idf_