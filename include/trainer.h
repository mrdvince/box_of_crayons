#ifndef trainer_h
#define trainer_h

#include <dirent.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <opencv2/opencv.hpp>
template <typename Trainloader, typename Validloader>
void trainer(torch::jit::script::Module net,
             torch::nn::Linear lin,
             Trainloader &data_loader,
             Validloader &valid_loader,
             torch::optim::Optimizer &optimizer,
             size_t dataset_size);
#endif