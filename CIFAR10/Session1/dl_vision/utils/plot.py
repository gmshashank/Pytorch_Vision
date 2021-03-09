import os
import matplotlib.pyplot as plt

# import seaborn as sns
# sns.set()
plt.style.use("dark_background")


def plot_metrics(train_metrics, test_metrics):
    (train_loss, train_accuracy) = train_metrics
    (test_loss, test_accuracy) = test_metrics

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Metrics")

    axs[0, 0].plot(train_loss)
    axs[0, 0].set_title("Training Loss")

    axs[1, 0].plot(train_accuracy)
    axs[1, 0].set_title("Train Accuracy")

    axs[0, 1].plot(test_loss)
    axs[0, 1].set_title("Test Loss")

    axs[1, 1].plot(test_accuracy)
    axs[1, 1].set_title("Test Accuracy")

    plt.tight_layout()

    plt.show()
    # plt.savefig("Metrics.png")
    return plt


def plot_misclassified(misclassified, max_count):
    print(f"Total Misclassified: {len(misclassified)}")
    fig = plt.figure(fig_size=(12, 10))
    fig.suptitle("25 Misclassifed Images")
    for idx, (image, prediction, target) in enumerate(misclassified["max_count"]):
        image, prediction, target = image.cpu(), prediction.cpu(), target.cpu()
        ax = fig.add_subplot(5, 5, 1 + idx)
        ax.axis("off")
        ax.set_title(
            f"target {target.item()} \nprediction {prediction.item()}", fontsize=11
        )
        ax.imshow(image.squeeze())
    plt.show()
