import torch as K

from lalnets2.gar import acolPool
from lalnets2.utils import calculate_cl_acc, to_numpy

def train(model, data_loaders, optimizer, L, U, nb_epoch=20, device="cuda"):
    
    for epoch in range(nb_epoch): 
        
        running_loss = 0.0
        running_reg = 0.0
        batches = 0.

        correct_Y = 0.
        correct_S = 0.
        total = 0.

        # training
        model.train()
        for data in data_loaders[0]:
            # get the inputs
            X, T = data
            X = X.to(device)
            T = T.to(device)
            # create parent labels
            T_tilda = K.tensor(T < 5, dtype=K.long, device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            Y, _, Z, _ = model(X)
            loss = L(Y, T_tilda)
            reg = U(Z)

            total_loss = loss + reg
            total_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_reg += reg.item()
            batches += 1

        # testing
        model.eval()
        with K.no_grad():
            for data in data_loaders[1]:
                # get the inputs
                X, T = data
                X = X.to(device)
                T = T.to(device)
                # create parent labels
                T_tilda = K.tensor(T < 5, dtype=K.long, device=device)
                # 
                Y, _, _, S = model(X, return_latent=True)

                _, predicted_Y = K.max(Y.data, 1)
                _, predicted_S = K.max(S.data, 1)

                cl_acc = calculate_cl_acc(to_numpy(T), to_numpy(predicted_S), U.k*U.n_p, label_correction=True)

                total += T.size(0)
                correct_Y += (predicted_Y == T_tilda).sum().item()
                correct_S += cl_acc[0]*cl_acc[1]

        running_loss /= batches
        running_reg /= batches
        acc = 100 * (correct_Y / total)
        cl_acc = 100 * (correct_S / total)

        print('%d, loss: %.3f, reg: %.3f, acc = %.2f %%, c_acc = %.2f %%' % (epoch + 1, 
                                                                             running_loss, 
                                                                             running_reg, 
                                                                             acc, cl_acc))
    print('Finished Training')