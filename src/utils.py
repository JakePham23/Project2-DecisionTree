import os
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import export_graphviz
import graphviz

def plot_label_original(labels, output_path):
    # Đếm số lượng mỗi class trong tập train và test
    train_dist = labels.value_counts().sort_index()

    # Vẽ biểu đồ
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig.suptitle(f"Class Original")

    sns.barplot(x=train_dist.index, y=train_dist.values, ax=ax)
    ax.set_title('Class Original')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')

    plt.tight_layout()
    output_filename = f"class_original.png"
    output_path = output_path + "charts/"
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, output_filename)
    plt.savefig(save_path)
    print("Class original:")
    plt.show()

def plot_label_distribution(label_train, label_test, train_size, test_size, output_path):
    # Đếm số lượng mỗi class trong tập train và test
    train_dist = label_train.value_counts().sort_index()
    test_dist = label_test.value_counts().sort_index()

    # Vẽ biểu đồ
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Class Distribution (Train {int(train_size)}% / Test {int(test_size)}%)")

    sns.barplot(x=train_dist.index, y=train_dist.values, ax=axs[0])
    axs[0].set_title('Train Set')
    axs[0].set_xlabel('Class')
    axs[0].set_ylabel('Count')

    sns.barplot(x=test_dist.index, y=test_dist.values, ax=axs[1])
    axs[1].set_title('Test Set')
    axs[1].set_xlabel('Class')
    axs[1].set_ylabel('Count')

    plt.tight_layout()
    output_filename = f"class_distribution_{int(train_size)}_{int(test_size)}.png"
    output_path = output_path + "charts/"    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, output_filename)
    plt.savefig(save_path)
    print(f"Class distribution ({int(train_size)}/{int(test_size)})")
    plt.show()

def build_decision_tree(features_final, feature_train, label_train, train_size, test_size, output_path):
    # Khởi tạo mô hình Cây quyết định với tiêu chí entropy
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)

    # Huấn luyện mô hình trên tập train
    clf.fit(feature_train, label_train)

    # Xuất ra file .dot
    dot_data = export_graphviz(clf, out_file=None, 
                            feature_names=features_final.columns,
                            class_names=['No Disease', 'Disease'],
                            filled=True, rounded=True, 
                            special_characters=True)

    # Dùng graphviz để hiển thị
    graph = graphviz.Source(dot_data)
    # Tạo thư mục nếu chưa tồn tại
    output_path_dot = output_path + "trees/"
    os.makedirs(output_path_dot, exist_ok=True)
    graph.render(f"{output_path_dot}decision_tree_{train_size}_{test_size}")
    print(f"Decision tree {train_size}/{test_size}")
    display(graph)
    return clf

def evaluating_decision_tree_class(clf, feature_test, label_test, train_size, test_size, output_path):
    # Dự đoán trên tập test
    label_pred = clf.predict(feature_test)

    # Classification report
    report = classification_report(label_test, label_pred, target_names=["No Disease", "Disease"])

    # Tạo thư mục nếu chưa tồn tại
    output_path_report = output_path + "reports/"
    os.makedirs(output_path_report, exist_ok=True)
    with open(f"{output_path_report}/classification_report_{train_size}_{test_size}.txt", "w") as f:
        f.write(report)
    print(f"Classification Report ({train_size}/{test_size}) đã được lưu tại: {output_path_report}/classification_report_{train_size}_{test_size}.txt")
    # Tạo confusion matrix
    cm = confusion_matrix(label_test, label_pred)

    # Vẽ heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Depth={clf.tree_.max_depth}, {train_size}/{test_size} Split)')
    plt.tight_layout()

    # Lưu hình vào thư mục output
    # Tạo thư mục nếu chưa tồn tại
    output_path_matrix = output_path + "matrices/"
    os.makedirs(output_path_matrix, exist_ok=True)
    conf_matrix_path = os.path.join(output_path_matrix, f'confusion_matrix_{train_size}_{test_size}.png')
    plt.savefig(conf_matrix_path)
    plt.close()

    print(f"Confusion Matrix (Depth={clf.tree_.max_depth}, {train_size}/{test_size} Split) đã được lưu tại: {conf_matrix_path}")

def build_decision_tree_penguins(features_final, feature_train, label_train, train_size, test_size, output_path):
     # 1. Khởi tạo và huấn luyện
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf.fit(feature_train, label_train)

    # 2. Tạo danh sách tên lớp động
    cls_names = [str(c) for c in clf.classes_]

    # 3. Xuất .dot
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=features_final.columns,
        class_names=cls_names,
        filled=True,
        rounded=True,
        special_characters=True
    )

    # 4. Hiển thị và lưu ảnh
    graph = graphviz.Source(dot_data)
    output_path_dot = os.path.join(output_path, "trees")
    os.makedirs(output_path_dot, exist_ok=True)
    graph.render(
        filename=f"decision_tree_{train_size}_{test_size}",
        directory=output_path_dot,
        format="png",
        cleanup=True
    )
    print(f"Đã tạo decision tree tại {os.path.join(output_path_dot, f'decision_tree_{train_size}_{test_size}.png')}")

    # 5. Trả về mô hình
    return clf

def evaluating_decision_tree_class_penguins(clf, feature_test, label_test, train_size, test_size, output_path):
    # Predict
    label_pred = clf.predict(feature_test)

    # Get class names
    classes = sorted(label_test.unique())

    # Classification report
    report = classification_report(label_test, label_pred, target_names=classes)

    # Save report
    output_path_report = os.path.join(output_path, "reports")
    os.makedirs(output_path_report, exist_ok=True)
    with open(os.path.join(output_path_report, f"classification_report_{train_size}_{test_size}.txt"), "w") as f:
        f.write(f"=== Classification Report ({train_size}/{test_size}) ===\n")
        f.write(report)

    print(f"Classification report saved: {output_path_report}/classification_report_{train_size}_{test_size}.txt")

    # Confusion matrix
    cm = confusion_matrix(label_test, label_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix ({train_size}/{test_size})")
    plt.tight_layout()

    # Save matrix
    output_path_matrix = os.path.join(output_path, "matrices")
    os.makedirs(output_path_matrix, exist_ok=True)
    matrix_path = os.path.join(output_path_matrix, f"confusion_matrix_{train_size}_{test_size}.png")
    plt.savefig(matrix_path)
    plt.close()
    print(f"Confusion matrix saved: {matrix_path}")
