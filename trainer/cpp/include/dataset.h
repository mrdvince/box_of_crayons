#pragma once
#include <dirent.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

torch::Tensor read_data(std::string image_path);
torch::Tensor read_labels(int label);
std::vector<torch::Tensor> process_images(std::vector<std::string> list_of_images);
std::vector<torch::Tensor> process_labels(std::vector<int> list_of_labels);
std::pair<std::vector<std::string>, std::vector<int>> load_data_from_folder(std::vector<std::string> folder_names);

class CDataset : public torch::data::Dataset<CDataset> {
   private:
    std::vector<torch::Tensor> images, labels;
    size_t dataset_size;

   public:
    CDataset(std::vector<std::string> list_images, std::vector<int> list_labels) {
        images = process_images(list_images);
        labels = process_labels(list_labels);
        dataset_size = images.size();
    };
    // get item
    torch::data::Example<> get(size_t index) override {
        torch::Tensor image = images.at(index);
        torch::Tensor label = labels.at(index);
        return {image, label};
    };
    torch::optional<size_t> size() const override {
        return dataset_size;
    };
};