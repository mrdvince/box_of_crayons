#pragma once
#include <torch/torch.h>

#include <iostream>

template <typename Trainloader,typename Validloader>
void trainer(torch::jit::script::Module net,
             torch::nn::Linear lin,
             Trainloader &train_loader,
             Validloader &valid_loader,
             torch::optim::Optimizer &optimizer,
             size_t dataset_size);
//  https://stackoverflow.com/questions/495021/why-can-templates-only-be-implemented-in-the-header-file
#include "trainer.tpp"