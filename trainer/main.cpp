#include <stdio.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "dataset.h"
#include "trainer.h"
std::vector<std::string> get_image_folders(std::string path) {
    const char *PATH = path.c_str();
    DIR *dir = opendir(PATH);
    struct dirent *ent = readdir(dir);

    std::vector<std::string> folder_names;
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0) {
            folder_names.push_back(path + "/" + ent->d_name);
            // std::cout << (ent->d_name) << "\n";
        }
    }
    closedir(dir);
    return folder_names;
};

int main(int argc, const char *argv[]) {
    // load dataset
    std::string DATA_DIR = "/home/vinc3/Projects/box_of_crayons/data/";
    std::string TRAIN_DIR = DATA_DIR + "/train";
    std::string VAL_DIR = DATA_DIR + "/val";

    std::vector<std::string> train_folder_images = get_image_folders(TRAIN_DIR);
    std::vector<std::string> val_folder_images = get_image_folders(VAL_DIR);

    // train
    std::pair<std::vector<std::string>, std::vector<int>> train_images_labels = load_data_from_folder(train_folder_images);
    std::vector<std::string> list_train_images = train_images_labels.first;
    std::vector<int> list_train_labels = train_images_labels.second;

    // trainloader
    auto train_dataset = CDataset(list_train_images, list_train_labels)
                             .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
                             .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), 64);

    // validation
    std::pair<std::vector<std::string>, std::vector<int>> valid_images_labels = load_data_from_folder(val_folder_images);
    std::vector<std::string> list_valid_images = valid_images_labels.first;
    std::vector<int> list_valid_labels = valid_images_labels.second;

    // validloader
    auto valid_dataset = CDataset(list_valid_images, list_valid_labels)
                             .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
                             .map(torch::data::transforms::Stack<>());

    auto valid_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(valid_dataset), 64);

    torch::jit::script::Module module;
    if (argc == 1) {
        module = torch::jit::load("resnet50_without_last_layer.pt");

    } else {
        module = torch::jit::load(argv[1]);
    }

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << device << std::endl;
    
    torch::nn::Linear linear(2048, 24);
    torch::optim::Adam optimizer(linear->parameters(), torch::optim::AdamOptions(0.001));
    module.to(device);
    linear->to(device);

    size_t train_size = train_dataset.size().value();

    trainer(module, linear, train_loader, valid_loader, optimizer, train_size);
}
