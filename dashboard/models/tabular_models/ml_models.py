from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def get_model(name):
    if name == "RandomForest⭐":
        return RandomForestClassifier()
    elif name == "SVM":
        return SVC(probability=True)
    elif name == "XGBoost":
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif name == "DecisionTree":
        return DecisionTreeClassifier()
    elif name == "MLP":
        return "MLP"