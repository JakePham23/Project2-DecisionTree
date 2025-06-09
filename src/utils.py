import os
import matplotlib.pyplot as plt
import seaborn as sns
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
        
    save_path = os.path.join(output_path, output_filename)
    plt.savefig(save_path)
    plt.close()

    print(f"Biểu đồ đã được lưu tại: {save_path}")

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
        
    save_path = os.path.join(output_path, output_filename)
    plt.savefig(save_path)
    plt.close()

    print(f"Biểu đồ đã được lưu tại: {save_path}")

def build_decision_tree(features_final, feature_train, feature_test, label_train, label_test, train_size, test_size, output_path):
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
    graph.render(f"{output_path}/decision_tree_{train_size}_{test_size}")  # thay đổi theo tỷ lệ

    # Dự đoán trên tập test
    label_pred = clf.predict(feature_test)

    # Classification report
    print(f"=== Classification Report ({train_size}/{test_size}) ===")
    print(classification_report(label_test, label_pred, target_names=["No Disease", "Disease"]))

    # Tạo confusion matrix
    cm = confusion_matrix(label_test, label_pred)

    # Vẽ heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({train_size}/{test_size})')
    plt.tight_layout()

    # Lưu hình vào thư mục output
    conf_matrix_path = os.path.join(output_path, f'confusion_matrix_({train_size}_{test_size}).png')
    plt.savefig(conf_matrix_path)
    plt.show()

    print(f"Confusion matrix đã được lưu tại: {conf_matrix_path}")
