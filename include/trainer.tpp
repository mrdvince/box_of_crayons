#include <torch/torch.h>

torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

template <typename Trainloader, typename Validloader>
void trainer(torch::jit::script::Module net,
             torch::nn::Linear lin,
             Trainloader &train_loader,
             Validloader &valid_loader,
             torch::optim::Optimizer &optimizer,
             size_t dataset_size) {
    float valid_loss = 0.0;
    float valid_loss_min = std::numeric_limits<float>::infinity();
    float train_loss = 0.0;

    for (int i = 0; i < 25; i++) {
        for (auto &batch : *train_loader) {
            auto data = batch.data.to(device);;
            auto target = batch.target.squeeze().to(device);;
            data = data.to(torch::kF32);
            target = target.to(torch::kInt64);
            std::vector<torch::jit::IValue> input;
            input.push_back(data);
            optimizer.zero_grad();
            auto output = net.forward(input).toTensor();
            // For transfer learning
            output = output.view({output.size(0), -1});
            output = lin(output);
            auto loss = torch::nll_loss(torch::log_softmax(output, 1), target);
            loss.backward();
            optimizer.step();
            train_loss += loss.template item<float>();
        }
        net.eval();
        int correct = 0, total = 0;
        for (auto &batch : *valid_loader) {
            auto data = batch.data.to(device);;
            auto target = batch.target.squeeze().to(device);;
            data = data.to(torch::kF32);
            target = target.to(torch::kInt64);
            std::vector<torch::jit::IValue> input;
            input.push_back(data);
            auto output = net.forward(input).toTensor();

            output = output.view({output.size(0), -1});
            output = lin(output);
            auto loss = torch::nll_loss(torch::log_softmax(output, 1), target);
            valid_loss += loss.template item<float>();

            auto out_tuple = torch::max(output, 1);
            auto predicted = std::get<1>(out_tuple);
            total += target.size(0);
            correct += (predicted == target).sum().template item<int>();
        }

        train_loss = train_loss / dataset_size;
        valid_loss = valid_loss / dataset_size;

        std::cout << "Epoch: " << i
                  << "\tTraining Loss: " << train_loss
                  << "\tValidation Loss: " << valid_loss
                  << "\tValidation Accuracy: " << (100 * correct / total) << " %" << std::endl;

        // save model if validation loss has decreased
        if (valid_loss <= valid_loss_min) {
            std::cout << "Validation loss decreased from: " << valid_loss_min << " --> " << valid_loss << "\nSaving model ...";
            valid_loss_min = valid_loss;
            net.save("module_model.pt");
            torch::save(lin, "model_linear.pt");
        }
    }
}