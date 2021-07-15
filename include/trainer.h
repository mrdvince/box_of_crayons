#pragma once
#include <torch/torch.h>

#include <iostream>

template <typename Dataloader>
void trainer(torch::jit::script::Module net,
             torch::nn::Linear lin,
             Dataloader &data_loader,
             Dataloader &valid_loader,
             torch::optim::Optimizer &optimizer,
             size_t dataset_size);
//  https://stackoverflow.com/questions/495021/why-can-templates-only-be-implemented-in-the-header-file
#include "trainer.tpp"