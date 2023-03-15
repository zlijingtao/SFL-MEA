
def adversarial_attack(self, attack_option, attack_client = 0, e = None):
    total_succ = 0
    self.surrogate_model.local.eval()
    self.surrogate_model.cloud.eval()
    criterion = nn.CrossEntropyLoss()
    if attack_option == "PGD_target":
        
        iter_max = 10
        if e is None:
            e = 0.05
        for temporal in range(iter_max):
            org_target = temporal
            attack_target = (org_target + 5)%9
            # history_list.append(org_target)
            self.logger.debug(f"###Round {temporal} src_label: {org_target} target_label: {attack_target}")
            
            
            succ = 0
            image_arr = torch.load('saved_tensors/data100/class_'+ str(org_target) +'.pth.tar')
            
            fake_label = torch.LongTensor(1)
            fake_label[0] = attack_target
            fake_label = torch.autograd.Variable(fake_label.cuda(), requires_grad=False)
            org_label = torch.zeros(image_arr.shape[0])
            attack_label = torch.zeros(image_arr.shape[0])
            succ_label = torch.zeros(image_arr.shape[0])
            succ_iter = torch.zeros(image_arr.shape[0])

            diff_succ = 0.0
            diff_all  = 0.0

            for i in range(image_arr.shape[0]):
                org_image = torch.FloatTensor(1, 3, 32, 32)
                org_image[0] = image_arr[i]
                org_image = torch.autograd.Variable(org_image.cuda(), requires_grad=True)

                act = self.model.local_list[attack_client](org_image)
                output = self.model.cloud(act)
                # print(output)
                _, org_pred = output.topk(1, 1, True, True)
                org_pred = org_pred.data[0, 0]
                fake_image = org_image.clone()
                #modify the original image
                max_val = torch.max(org_image).item()
                min_val = torch.min(org_image).item()
                for iter in range(50):
                    # calculate gradient
                    grad = torch.zeros(1, 3, 32, 32).cuda()
                    fake_image = torch.autograd.Variable(fake_image.cuda(), requires_grad=True)
                    if fake_image.grad is not None:
                        fake_image.grad.zero_()
                    act = self.surrogate_model.local(fake_image)
                    output = self.surrogate_model.cloud(act)
                    loss = criterion(output, fake_label)
                    loss.backward()
                    #print(loss)
                    grad += torch.sign(fake_image.grad)

                    fake_image = fake_image - grad * e
                    fake_image[fake_image > max_val] = max_val
                    fake_image[fake_image < min_val] = min_val
                    act = self.model.local_list[attack_client](fake_image)
                    output = self.model.cloud(act)
                    _, fake_pred = output.topk(1, 1, True, True)
                    fake_pred = fake_pred[0, 0]
                    
                    if fake_label.item() == fake_pred.item() or iter == 49:
                        # print(fake_pred.item(), fake_label.item())
                        attack_pred_list = []
                        act = self.surrogate_model.local(fake_image)
                        output = self.surrogate_model.cloud(act)
                        _, attack_pred = output.topk(1, 1, True, True)
                        attack_pred_list.append(attack_pred.data[0, 0].item())

                        # if (i + 1) % 20 == 0:
                        #     print(str(i) + '\torg prediction: ' + str(org_pred.item()) + '\tfake prediction: ' + str(fake_pred.item()) +
                        #     '\tattack_pred: ' + str(attack_pred_list) + '\te: ' + str(e) + '\titer: ' + str(iter) + '\tsucc: ' + str(succ))

                        org_label[i] = org_pred.item()
                        attack_label[i] = fake_pred.item()
                        succ_iter[i] = iter + 1
                        
                        diff = torch.sum((org_image - fake_image) ** 2).item()
                        diff_all += diff

                        if fake_label.item() == fake_pred:
                            diff_succ += diff
                            succ += 1
                            succ_label[i] = 1
                        break


            diff_all /= (1.0 * image_arr.shape[0])
            if succ > 0:
                diff_succ /= (1.0 * succ)
            
            str_log = 'src: ' + str(org_target) + '\ttar: ' + str(attack_target)+ '\ttotal: ' + str(i + 1) + '\tsuccess: ' + str(succ) + '\tavg iter: ' + str(torch.mean(succ_iter).item()) + '\tdiff_suc: ' + str(diff_succ) + '\tdif_total: ' + str(diff_all) 
            self.logger.debug(str_log)
            total_succ += succ
        self.logger.debug(f"Total succeed times {total_succ}.")
    elif attack_option == "FGSM":
        if e is None:
            e = 0.1
        iter_max = 10
        for temporal in range(iter_max):
            org_target = temporal
            image_arr = torch.load('saved_tensors/data100/class_'+ str(org_target) +'.pth.tar')
            succ = 0
            diff_succ = 0.0
            diff_all  = 0.0
            fake_label = torch.LongTensor(1)
            fake_label[0] = org_target
            fake_label = torch.autograd.Variable(fake_label.cuda(), requires_grad=False)
            org_label = torch.zeros(image_arr.shape[0])
            attack_label = torch.zeros(image_arr.shape[0])
            succ_label = torch.zeros(image_arr.shape[0])
            succ_iter = torch.zeros(image_arr.shape[0])
            for i in range(image_arr.shape[0]):
                #for i in range(2):
                org_image = torch.FloatTensor(1, 3, 32, 32)
                org_image[0] = image_arr[i]
                org_image = torch.autograd.Variable(org_image.cuda(), requires_grad=True)

                act = self.model.local_list[attack_client](org_image)
                output = self.model.cloud(act)
                _, org_pred = output.topk(1, 1, True, True)
                org_pred = org_pred.data[0, 0]

                _, org_pred = output.topk(1, 1, True, True)
                org_pred = org_pred.data[0, 0]
                
                fake_image = org_image.clone()

                #modify the original image
                max_val = torch.max(org_image).item()
                min_val = torch.min(org_image).item()
                
                # calculate gradient
                grad = torch.zeros(1, 3, 32, 32).cuda()
                fake_image = torch.autograd.Variable(fake_image.cuda(), requires_grad=True)

                if fake_image.grad is not None:
                    fake_image.grad.zero_()
                act = self.surrogate_model.local(fake_image)
                output = self.surrogate_model.cloud(act)
                loss = criterion(output, fake_label)
                loss.backward()
                grad -= torch.sign(fake_image.grad)

                fake_image = fake_image - grad * e # e is epsilon in FGSM:  https://arxiv.org/pdf/1706.06083.pdf
                fake_image[fake_image > max_val] = max_val
                fake_image[fake_image < min_val] = min_val
                act = self.model.local_list[attack_client](fake_image)
                output = self.model.cloud(act)

                _, fake_pred = output.topk(1, 1, True, True)
                fake_pred = fake_pred.data[0, 0]

                attack_pred_list = []
                act = self.surrogate_model.local(fake_image)
                output = self.surrogate_model.cloud(act)
                _, attack_pred = output.topk(1, 1, True, True)
                attack_pred_list.append(attack_pred.data[0, 0].item())

                if (i + 1) % 20 == 0:
                    print(str(i) + '\torg prediction: ' + str(org_pred.item()) + '\tfake prediction: ' + str(fake_pred.item()) +
                        '\tattack_pred: ' + str(attack_pred_list) + '\te: ' + str(e) + '\tsucc: ' + str(succ))

                org_label[i] = org_pred.item()
                attack_label[i] = fake_pred.item()
                
                diff = torch.sum((org_image - fake_image) ** 2).item()
                diff_all += diff

                if fake_label.item() != fake_pred:
                    diff_succ += diff
                    succ += 1
                    succ_label[i] = 1
                    # break


            diff_all /= (1.0 * image_arr.shape[0])
            if succ > 0:
                diff_succ /= (1.0 * succ)
            
            str_log = 'src: ' + str(org_target) + '\ttotal: ' + str(i + 1) + '\tsuccess: ' + str(succ) + '\tdiff_suc: ' + str(diff_succ) + '\tdif_total: ' + str(diff_all) 
            # print(str_log)
            self.logger.debug(str_log)
            total_succ += succ
        self.logger.debug(f"Total succeed times {total_succ}.")
    elif attack_option == "PGD":
        if e is None:
            e = 0.02
        for temporal in range(self.num_class):
            org_target = temporal
            image_arr = torch.load('saved_tensors/data100/class_'+ str(org_target) +'.pth.tar')
            succ = 0
            diff_succ = 0.0
            diff_all  = 0.0
            fake_label = torch.LongTensor(1)
            fake_label[0] = org_target
            fake_label = torch.autograd.Variable(fake_label.cuda(), requires_grad=False)
            org_label = torch.zeros(image_arr.shape[0])
            attack_label = torch.zeros(image_arr.shape[0])
            succ_label = torch.zeros(image_arr.shape[0])
            succ_iter = torch.zeros(image_arr.shape[0])
            for i in range(image_arr.shape[0]):
            #for i in range(2):
                org_image = torch.FloatTensor(1, 3, 32, 32)
                org_image[0] = image_arr[i]
                # print(image_arr[i])
                org_image = torch.autograd.Variable(org_image.cuda(), requires_grad=True)

                act = self.surrogate_model.local(org_image)
                output = self.surrogate_model.cloud(act)

                # activation = nn.LogSoftmax(1)

                # output = activation(output)
                
                _, org_pred = output.topk(1, 1, True, True)
                org_pred = org_pred.data[0, 0]
                # if i < 50:
                #     print(org_pred)
                fake_image = org_image.clone()

                #modify the original image
                max_val = torch.max(org_image).item()
                min_val = torch.min(org_image).item()
                for iter in range(50): # PGD: 
                    # calculate gradient
                    grad = torch.zeros(1, 3, 32, 32).cuda()
                    fake_image = torch.autograd.Variable(fake_image.cuda(), requires_grad=True)

                    if fake_image.grad is not None:
                        fake_image.grad.zero_()
                    act = self.surrogate_model.local(fake_image)
                    output = self.surrogate_model.cloud(act)
                    loss = criterion(output, fake_label)
                    loss.backward()
                    #print(loss)
                    grad -= torch.sign(fake_image.grad)

                    fake_image = fake_image - grad * e # e is alpha in PGD:  https://arxiv.org/pdf/1706.06083.pdf
                    fake_image[fake_image > max_val] = max_val
                    fake_image[fake_image < min_val] = min_val
                    act = self.surrogate_model.local(fake_image)
                    output = self.surrogate_model.cloud(act)

                    _, fake_pred = output.topk(1, 1, True, True)
                    fake_pred = fake_pred.data[0, 0]

                    if fake_label.item() != fake_pred.item() or iter == 49:
                        # print(fake_pred.item(), fake_label.item())
                        attack_pred_list = []

                        act = self.surrogate_model.local(fake_image)
                        output = self.surrogate_model.cloud(act)
                        _, attack_pred = output.topk(1, 1, True, True)
                        attack_pred_list.append(attack_pred.data[0, 0].item())

                        if (i + 1) % 20 == 0:
                            print(str(i) + '\torg prediction: ' + str(org_pred.item()) + '\tfake prediction: ' + str(fake_pred.item()) +
                            '\tattack_pred: ' + str(attack_pred_list) + '\te: ' + str(e) + '\titer: ' + str(iter) + '\tsucc: ' + str(succ))

                        org_label[i] = org_pred.item()
                        attack_label[i] = fake_pred.item()
                        succ_iter[i] = iter + 1
                        
                        diff = torch.sum((org_image - fake_image) ** 2).item()
                        diff_all += diff

                        if fake_label.item() != fake_pred:
                            diff_succ += diff
                            succ += 1
                            succ_label[i] = 1
                        break


            diff_all /= (1.0 * image_arr.shape[0])
            if succ > 0:
                diff_succ /= (1.0 * succ)
            #print('total: ' + str(i + 1) + '\tsuccess: ' + str(succ) + '\tavg iter: ' + str(torch.mean(succ_iter).item()))
            
            str_log = 'src: ' + str(org_target) + '\ttar: ' + '\ttotal: ' + str(i + 1) + '\tsuccess: ' + str(succ) + '\tavg iter: ' + str(torch.mean(succ_iter).item()) + '\tdiff_suc: ' + str(diff_succ) + '\tdif_total: ' + str(diff_all) 
            self.logger.debug(str_log)
            total_succ += succ
        self.logger.debug(f"Total succeed times {total_succ}.")

