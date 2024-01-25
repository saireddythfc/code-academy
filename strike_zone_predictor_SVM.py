import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()

def viz(player):

    #print(player.type.unique())

    player["type"] = player["type"].map({'S': 1, 'B': 0})
    #print(player["type"])

    #print(player["plate_x"])

    player = player.dropna(subset = ["type", "plate_x", "plate_z","strikes"])
    #print(player["type"])

    training_set, validation_set = train_test_split(player, random_state = 1)

    classifier = SVC(kernel = 'rbf', gamma = 3, C = 1)
    classifier.fit(training_set[["plate_x", "plate_z","strikes"]], training_set["type"])

    plt.scatter(x = player["plate_x"], y = player["plate_z"], c = player["type"], cmap = plt.cm.coolwarm, alpha = 0.25)

    #draw_boundary(ax, classifier)
    ax.set_ylim(-2, 6)
    ax.set_xlim(-3, 3)
    plt.show()

    acc = classifier.score(validation_set[["plate_x", "plate_z","strikes"]], validation_set["type"])
    print(acc)

#viz(aaron_judge)
#viz(jose_altuve)
viz(david_ortiz)

