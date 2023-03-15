

def gen_ir(self, val_single_loader, local_model, img_folder="./tmp", intermed_reps_folder="./tmp", all_label=True,
            select_label=0, attack_from_later_layer=-1, attack_option = "MIA"):
    """
    Generate (Raw Input - Intermediate Representation) Pair for Training of the AutoEncoder
    """

    # switch to evaluate mode
    local_model.eval()
    file_id = 0
    for i, (input, target) in enumerate(val_single_loader):
        # input = input.cuda(async=True)
        input = input.cuda()
        target = target.item()
        if not all_label:
            if target != select_label:
                continue

        img_folder = os.path.abspath(img_folder)
        intermed_reps_folder = os.path.abspath(intermed_reps_folder)
        if not os.path.isdir(intermed_reps_folder):
            os.makedirs(intermed_reps_folder)
        if not os.path.isdir(img_folder):
            os.makedirs(img_folder)

        # compute output
        with torch.no_grad():
            ir = local_model(input)
        
        if self.confidence_score:
            self.model.cloud.eval()
            ir = self.model.cloud(ir)
            if self.arch == "resnet18" or self.arch == "resnet34" or "mobilenetv2" in self.arch:
                ir = F.avg_pool2d(ir, 4)
                ir = ir.view(ir.size(0), -1)
                ir = self.classifier(ir)
            elif self.arch == "resnet20" or self.arch == "resnet32":
                ir = F.avg_pool2d(ir, 8)
                ir = ir.view(ir.size(0), -1)
                ir = self.classifier(ir)
            else:
                ir = ir.view(ir.size(0), -1)
                ir = self.classifier(ir)
        
        if attack_from_later_layer > -1 and (not self.confidence_score):
            self.model.cloud.eval()

            activation_4 = {}

            def get_activation_4(name):
                def hook(model, input, output):
                    activation_4[name] = output.detach()

                return hook

            with torch.no_grad():
                activation_4 = {}
                count = 0
                for name, m in self.model.cloud.named_modules():
                    if attack_from_later_layer == count:
                        m.register_forward_hook(get_activation_4("ACT-{}".format(name)))
                        valid_key = "ACT-{}".format(name)
                        break
                    count += 1
                output = self.model.cloud(ir)
            try:
                ir = activation_4[valid_key]
            except:
                print("cannot attack from later layer, server-side model is empty or does not have enough layers")
        ir = ir.float()

        if "truncate" in attack_option:
            try:
                percentage_left = int(attack_option.split("truncate")[1])
            except:
                print("auto extract percentage fail. Use default percentage_left = 20")
                percentage_left = 20
            ir = prune_top_n_percent_left(ir, percentage_left)

        inp_img_path = "{}/{}.jpg".format(img_folder, file_id)
        out_tensor_path = "{}/{}.pt".format(intermed_reps_folder, file_id)
        if DENORMALIZE_OPTION:
            input = denormalize(input, self.dataset)
        save_image(input, inp_img_path)
        torch.save(ir.cpu(), out_tensor_path)
        file_id += 1
    print("Overall size of Training/Validation Datset for AE is {}: {}".format(int(file_id * 0.9),
                                                                                int(file_id * 0.1)))


def train_generator(self, num_query, nz, data_helper = None, resume = False, discriminator_option = False, pred_option = False):
    
    lr_G = 1e-4
    lr_C = 1e-4
    d_iter = 5
    
    if pred_option:
        num_steps = num_query // 100 // (1 + d_iter) # train g_iter + d_iter times
    else:
        num_steps = num_query // 100 # train once
    steps = sorted([int(step * num_steps) for step in [0.1, 0.3, 0.5]])
    scale = 3e-1
    

    D_w = self.noise_w

    if self.dataset == "cifar10":
        D_w = D_w * 10
    
    # Define Discriminator, ways to suppress D: reduce_learning rate, increase label_smooth, enable dropout, reduce Resblock_repeat
    label_smoothing = 0.1
    gan_discriminator = architectures.discriminator((3, 32, 32), True, resblock_repeat = 0, dropout = True)
    optimizer_C = torch.optim.Adam(gan_discriminator.parameters(), lr=lr_C )
    scheduler_C = torch.optim.lr_scheduler.MultiStepLR(optimizer_C, steps, scale)
    
    optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G )
    scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, steps, scale)
    
    train_output_path = self.save_dir + "generator_train"
    if os.path.isdir(train_output_path):
        rmtree(train_output_path)
    os.makedirs(train_output_path)
    # best_acc = 0
    # acc_list = []
    if resume:
        G_state_dict = torch.load(self.save_dir + f"/checkpoint_generator_200.tar")
        self.generator.load_state_dict(G_state_dict)
        self.generator.cuda()
        self.generator.eval()

        z = torch.randn((10, nz)).cuda()
        for i in range(self.num_class):
            labels = i * torch.ones([10, ]).long().cuda()
            #Get fake image from generator
            fake = self.generator(z, labels) # pre_x returns the output of G before applying the activation

            imgGen = fake.clone()
            if DENORMALIZE_OPTION:
                imgGen = denormalize(imgGen, self.dataset)
            if not os.path.isdir(train_output_path + "/{}".format(num_steps)):
                os.mkdir(train_output_path + "/{}".format(num_steps))
            torchvision.utils.save_image(imgGen, train_output_path + '/{}/out_{}.jpg'.format(num_steps,"final_label{}".format(i)))
    else:
        self.generator.cuda()
        self.generator.train()
        gan_discriminator.cuda()
        gan_discriminator.train()

        if data_helper is not None and discriminator_option:
            data_iterator = iter(data_helper)
        else:
            data_iterator = None

        criterion = torch.nn.CrossEntropyLoss()
        BCE_loss = torch.nn.BCELoss()
        
        bc_losses = AverageMeter()
        bc_losses_gan = AverageMeter()
        ce_losses = AverageMeter()
        g_losses = AverageMeter()



        max_acc = 0
        max_fidelity = 0

        

        for i in range(1, num_steps + 1):
            

            if pred_option:
                if i % 10 == 0:
                    self.suro_scheduler.step()
                    val_accu, fidel_score = self.steal_test()
                    if val_accu > max_acc:
                        max_acc = val_accu
                    if fidel_score > max_fidelity:
                        max_fidelity = fidel_score
                    self.logger.debug("Step: {}, val_acc: {}, val_fidelity: {}".format(i, val_accu, fidel_score))
                

            if i % 10 == 0:
                bc_losses = AverageMeter()
                bc_losses_gan = AverageMeter()
                ce_losses = AverageMeter()
                g_losses = AverageMeter()
                

            
            scheduler_G.step()
            scheduler_C.step()

            #Sample Random Noise
            z = torch.randn((100, nz)).cuda()
            
            B = 50

            labels_l = torch.randint(low=0, high=self.num_class, size = [B, ]).cuda()
            labels_r = copy.deepcopy(labels_l).cuda()
            labels = torch.stack([labels_l, labels_r]).view(-1)
            zero_label = torch.zeros((50, )).cuda() 
            
            one_label = torch.ones((50, )).cuda() 

            '''Train Generator'''
            optimizer_G.zero_grad()
            
            #Get fake image from generator
            fake = self.generator(z, labels) # pre_x returns the output of G before applying the activation
            
            if i % 10 == 0:
                imgGen = fake.clone()
                if DENORMALIZE_OPTION:
                    imgGen = denormalize(imgGen, self.dataset)
                if not os.path.isdir(train_output_path + "/train"):
                    os.mkdir(train_output_path + "/train")
                torchvision.utils.save_image(imgGen, train_output_path + '/train/out_{}.jpg'.format(i * 100 + 100))
            
            
            # with torch.no_grad(): 
            output = self.model.local_list[0](fake)

            output = self.model.cloud(output)

            s_output = self.surrogate_model.local(fake)

            s_output = self.surrogate_model.cloud(s_output)
            
            # Diversity-aware regularization https://sites.google.com/view/iclr19-dsgan/
            
            g_noise_out_dist = torch.mean(torch.abs(fake[:B, :] - fake[B:, :]))
            g_noise_z_dist = torch.mean(torch.abs(z[:B, :] - z[B:, :]).view(B,-1),dim=1)
            g_noise = torch.mean( g_noise_out_dist / g_noise_z_dist ) * self.noise_w


            if not pred_option:
                #Cross Entropy Loss
                ce_loss = criterion(output, labels)

                loss = ce_loss - g_noise
                
            else:
                # ce_loss = criterion(output, labels) - 0.5* student_Loss(s_output, output)
                # ce_loss = - 1* student_Loss(s_output, output)
                ce_loss = criterion(output, labels)
                # ce_loss = - 0.1* student_Loss(s_output, output)
                # F.kl_div(s_output, output.detach(), reduction='none').sum(dim = 1).view(-1, m + 1) 
                loss = ce_loss - g_noise
            ## Discriminator Loss
            if data_helper is not None and discriminator_option:
                d_out = gan_discriminator(fake)
                bc_loss_gan = D_w * BCE_loss(d_out.reshape(-1), one_label)
                loss += bc_loss_gan
                bc_losses_gan.update(bc_loss_gan.item(), 100)
            
            loss.backward()

            optimizer_G.step()

            ce_losses.update(ce_loss.item(), 100)
            
            g_losses.update(g_noise.item(), 100)

            '''Train surrogate model (to match teacher and student, assuming having prediction access)'''
            if pred_option:
                
                for _ in range(d_iter):
                    z = torch.randn((100, nz)).cuda()
                    labels = torch.randint(low=0, high=self.num_class, size = [100, ]).cuda()
                    fake = self.generator(z, labels).detach()
                    

                    with torch.no_grad(): 
                        t_out = self.model.local_list[0](fake)
                        t_out = self.model.cloud(t_out)
                        t_out = F.log_softmax(t_out, dim=1).detach()
                    
                    self.suro_optimizer.zero_grad()
                    s_out = self.surrogate_model.local(fake)
                    s_out = self.surrogate_model.cloud(s_out)

                    loss_S = student_Loss(s_out, t_out) 
                    # loss_S = student_Loss(s_out, t_out) + criterion(s_out, labels)
                    loss_S.backward()
                    self.suro_optimizer.step()

            '''Train Discriminator (tell real/fake, using data_helper)'''
            if data_helper is not None and discriminator_option:
                try:
                    images, _ = next(data_iterator)
                    if images.size(0) != 100:
                        data_iterator = iter(data_helper)
                        images, _ = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(data_helper)
                    images, _ = next(data_iterator)
                
                images = images.cuda()

                d_input = torch.cat((fake.detach(), images), dim = 0)

                d_label =  torch.cat((zero_label, one_label - label_smoothing), dim = 0)

                optimizer_C.zero_grad()

                d_output = gan_discriminator(d_input)
                
                bc_loss = BCE_loss(d_output.reshape(-1), d_label)
                
                bc_loss.backward()
                
                optimizer_C.step()

                bc_losses.update(bc_loss.item(), 100)

            # Log Results
            if i % 10 == 0:
                self.logger.debug(f'Train step: {i}\t CE_Loss: {ce_losses.avg:.10f} diversity_Loss: {g_losses.avg:.10f} bc_losses (G): {bc_losses_gan.avg: .10f}  bc_losses (D)): {bc_losses.avg: .10f}')
        
        if pred_option:
            self.logger.debug("Best perform model, val_acc: {}, fidel_score: {}".format(max_acc, max_fidelity))
        self.logger.debug(f'End of Training: \t CE_Loss: {ce_losses.avg:.10f} diversity_Loss: {g_losses.avg:.10f} bc_losses (G): {bc_losses_gan.avg: .10f}  bc_losses (D)): {bc_losses.avg: .10f}')

        self.generator.cuda()
        self.generator.eval()

        z = torch.randn((10, nz)).cuda()
        for i in range(self.num_class):
            labels = i * torch.ones([10, ]).long().cuda()
            #Get fake image from generator
            fake = self.generator(z, labels) # pre_x returns the output of G before applying the activation

            imgGen = fake.clone()
            if DENORMALIZE_OPTION:
                imgGen = denormalize(imgGen, self.dataset)
            if not os.path.isdir(train_output_path + "/{}".format(num_steps)):
                os.mkdir(train_output_path + "/{}".format(num_steps))
            torchvision.utils.save_image(imgGen, train_output_path + '/{}/out_{}.jpg'.format(num_steps,"final_label{}".format(i)))


def MIA_attack(self, num_epochs, attack_option="MIA", collude_client=1, target_client=0, noise_aware=False,
                loss_type="MSE", attack_from_later_layer=-1, MIA_optimizer = "Adam", MIA_lr = 1e-3):
    
    # setup gan_adv regularizer
    self.gan_AE_activation = "sigmoid"
    self.gan_AE_type = gan_AE_type
    self.gan_loss_type = gan_loss_type
    self.alpha2 = regularization_strength  # set to 1~10
    
    attack_option = attack_option
    MIA_optimizer = MIA_optimizer
    MIA_lr = MIA_lr
    attack_batchsize = 32
    attack_num_epochs = num_epochs
    model_log_file = self.save_dir + '/{}_attack_{}_{}.log'.format(attack_option, collude_client, target_client)
    logger = setup_logger('{}_{}to{}_attack_logger'.format(str(self.save_dir), collude_client, target_client),
                            model_log_file, level=logging.DEBUG)
    # pass
    image_data_dir = self.save_dir + "/img"
    tensor_data_dir = self.save_dir + "/img"

    # Clear content of image_data_dir/tensor_data_dir
    if os.path.isdir(image_data_dir):
        rmtree(image_data_dir)
    if os.path.isdir(tensor_data_dir):
        rmtree(tensor_data_dir)

    if self.dataset == "cifar100":
        val_single_loader, _, _ = get_cifar100_testloader(batch_size=1, num_workers=4, shuffle=False)
    elif self.dataset == "cifar10":
        val_single_loader, _, _ = get_cifar10_testloader(batch_size=1, num_workers=4, shuffle=False)
    elif self.dataset == "svhn":
        val_single_loader, _, _ = get_SVHN_testloader(batch_size=1, num_workers=4, shuffle=False)
    elif self.dataset == "mnist":
        _, val_single_loader = get_mnist_bothloader(batch_size=1, num_workers=4, shuffle=False)
    elif self.dataset == "fmnist":
        _, val_single_loader = get_fmnist_bothloader(batch_size=1, num_workers=4, shuffle=False)
    elif self.dataset == "femnist":
        _, val_single_loader = get_femnist_bothloader(batch_size=1, num_workers=4, shuffle=False)
    elif self.dataset == "facescrub":
        _, val_single_loader = get_facescrub_bothloader(batch_size=1, num_workers=4, shuffle=False)
    elif self.dataset == "tinyimagenet":
        _, val_single_loader = get_tinyimagenet_bothloader(batch_size=1, num_workers=4, shuffle=False)
    elif self.dataset == "imagenet":
        val_single_loader = get_imagenet_testloader(batch_size=1, num_workers=4, shuffle=False)
    attack_path = self.save_dir + '/{}_attack_{}to{}'.format(attack_option, collude_client, target_client)
    if not os.path.isdir(attack_path):
        os.makedirs(attack_path)
        os.makedirs(attack_path + "/train")
        os.makedirs(attack_path + "/test")
        os.makedirs(attack_path + "/tensorboard")
        os.makedirs(attack_path + "/sourcecode")
    train_output_path = "{}/train".format(attack_path)
    test_output_path = "{}/test".format(attack_path)
    tensorboard_path = "{}/tensorboard/".format(attack_path)
    model_path = "{}/model.pt".format(attack_path)
    path_dict = {"model_path": model_path, "train_output_path": train_output_path,
                    "test_output_path": test_output_path, "tensorboard_path": tensorboard_path}

    if ("MIA" in attack_option) and ("MIA_mf" not in attack_option):
        logger.debug("Generating IR ...... (may take a while)")

        if collude_client == 0:
            self.gen_ir(val_single_loader, self.f, image_data_dir, tensor_data_dir,
                        attack_from_later_layer=attack_from_later_layer, attack_option = attack_option)
        elif collude_client == 1:
            self.gen_ir(val_single_loader, self.c, image_data_dir, tensor_data_dir,
                        attack_from_later_layer=attack_from_later_layer, attack_option = attack_option)
        elif collude_client > 1:
            self.gen_ir(val_single_loader, self.model.local_list[collude_client], image_data_dir, tensor_data_dir,
                        attack_from_later_layer=attack_from_later_layer, attack_option = attack_option)
        for filename in os.listdir(tensor_data_dir):
            if ".pt" in filename:
                sampled_tensor = torch.load(tensor_data_dir + "/" + filename)
                input_nc = sampled_tensor.size()[1]
                try:
                    input_dim = sampled_tensor.size()[2]
                except:
                    print("Extract input dimension fialed, set to 0")
                    input_dim = 0
                break

        if self.gan_AE_type == "custom":
            decoder = architectures.custom_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32,
                                                activation=self.gan_AE_activation).cuda()
        elif self.gan_AE_type == "custom_bn":
            decoder = architectures.custom_AE_bn(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32,
                                                    activation=self.gan_AE_activation).cuda()
        elif self.gan_AE_type == "complex":
            decoder = architectures.complex_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32,
                                                activation=self.gan_AE_activation).cuda()
        elif self.gan_AE_type == "complex_plus":
            decoder = architectures.complex_plus_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                    output_dim=32, activation=self.gan_AE_activation).cuda()
        elif self.gan_AE_type == "complex_res":
            decoder = architectures.complex_res_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                    output_dim=32, activation=self.gan_AE_activation).cuda()
        elif self.gan_AE_type == "complex_resplus":
            decoder = architectures.complex_resplus_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                        output_dim=32, activation=self.gan_AE_activation).cuda()
        elif "complex_resplusN" in self.gan_AE_type:
            try:
                N = int(self.gan_AE_type.split("complex_resplusN")[1])
            except:
                print("auto extract N from complex_resplusN failed, set N to default 2")
                N = 2
            decoder = architectures.complex_resplusN_AE(N = N, input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                            output_dim=32, activation=self.gan_AE_activation).cuda()
        elif "complex_normplusN" in self.gan_AE_type:
            try:
                afterfix = self.gan_AE_type.split("complex_normplusN")[1]
                N = int(afterfix.split("C")[0])
                internal_C = int(afterfix.split("C")[1])
            except:
                print("auto extract N from complex_normplusN failed, set N to default 2")
                N = 0
                internal_C = 64
            decoder = architectures.complex_normplusN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                        input_dim=input_dim, output_dim=32,
                                                        activation=self.gan_AE_activation).cuda()
        
        elif "conv_normN" in self.gan_AE_type:
            try:
                afterfix = self.gan_AE_type.split("conv_normN")[1]
                N = int(afterfix.split("C")[0])
                internal_C = int(afterfix.split("C")[1])
            except:
                print("auto extract N from conv_normN failed, set N to default 2")
                N = 0
                internal_C = 64
            decoder = architectures.conv_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                        input_dim=input_dim, output_dim=32,
                                                        activation=self.gan_AE_activation).cuda()

        elif "res_normN" in self.gan_AE_type:
            try:
                afterfix = self.gan_AE_type.split("res_normN")[1]
                N = int(afterfix.split("C")[0])
                internal_C = int(afterfix.split("C")[1])
            except:
                print("auto extract N from res_normN failed, set N to default 2")
                N = 0
                internal_C = 64
            decoder = architectures.res_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                        input_dim=input_dim, output_dim=32,
                                                        activation=self.gan_AE_activation).cuda()
        
        elif "TB_normplusN" in self.gan_AE_type:
            try:
                afterfix = self.gan_AE_type.split("TB_normplusN")[1]
                N = int(afterfix.split("C")[0])
                internal_C = int(afterfix.split("C")[1])
            except:
                print("auto extract N from TB_normplusN failed, set N to default 0")
                N = 0
                internal_C = 64
            decoder = architectures.TB_normplusN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                        input_dim=input_dim, output_dim=32,
                                                        activation=self.gan_AE_activation).cuda()
        elif self.gan_AE_type == "simple":
            decoder = architectures.simple_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32,
                                                activation=self.gan_AE_activation).cuda()
        elif self.gan_AE_type == "simple_bn":
            decoder = architectures.simple_AE_bn(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32,
                                                    activation=self.gan_AE_activation).cuda()
        elif self.gan_AE_type == "simplest":
            decoder = architectures.simplest_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32,
                                                activation=self.gan_AE_activation).cuda()
        else:
            raise ("No such GAN AE type.")

        if self.measure_option:
            noise_input = torch.randn([1, input_nc, input_dim, input_dim])
            device = next(decoder.parameters()).device
            noise_input = noise_input.to(device)
            macs, num_param = profile(decoder, inputs=(noise_input,))
            self.logger.debug(
                "{} Decoder Model's Mac and Param are {} and {}".format(self.gan_AE_type, macs, num_param))
            '''Uncomment below to also get decoder's inference and training time overhead.'''
            # decoder.cpu()
            # noise_input = torch.randn([128, input_nc, input_dim, input_dim])
            # with torch.no_grad():
            #     _ = decoder(noise_input)
            #     start_time = time.time()
            #     for _ in range(500):  # CPU warm up
            #         _ = decoder(noise_input)
            #     lapse_cpu_decoder = (time.time() - start_time) / 500
            # self.logger.debug("Decoder Model's Inference time on CPU is {}".format(lapse_cpu_decoder))

            # criterion = torch.nn.MSELoss()
            # noise_reconstruction = torch.randn([128, 3, 32, 32])
            # reconstruction = decoder(noise_input)

            # r_loss = criterion(reconstruction, noise_reconstruction)
            # r_loss.backward()
            # lapse_cpu_decoder_train = 0
            # for _ in range(500):  # CPU warm up
            #     reconstruction = decoder(noise_input)
            #     r_loss = criterion(reconstruction, noise_reconstruction)
            #     start_time = time.time()
            #     r_loss.backward()
            #     lapse_cpu_decoder_train += (time.time() - start_time)
            # lapse_cpu_decoder_train = lapse_cpu_decoder_train / 500
            # del r_loss, reconstruction, noise_input
            # self.logger.debug("Decoder Model's Train time on CPU is {}".format(lapse_cpu_decoder_train))
            # decoder.cuda()

        '''Setting attacker's learning algorithm'''
        # optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
        if MIA_optimizer == "Adam":
            optimizer = torch.optim.Adam(decoder.parameters(), lr=MIA_lr)
        elif MIA_optimizer == "SGD":
            optimizer = torch.optim.SGD(decoder.parameters(), lr=MIA_lr)
        else:
            raise("MIA optimizer {} is not supported!".format(MIA_optimizer))
        # Construct a dataset for training the decoder
        trainloader, testloader = apply_transform(attack_batchsize, image_data_dir, tensor_data_dir)

        # Do real test on target's client activation (and test with target's client ground-truth.)
        sp_testloader = apply_transform_test(1, self.save_dir + "/save_activation_client_{}_epoch_{}".format(
            target_client, self.n_epochs), self.save_dir + "/save_activation_client_{}_epoch_{}".format(target_client,
                                                                                                        self.n_epochs))

        if "gan_adv_noise" in self.regularization_option and noise_aware:
            print("create a second decoder") # Avoid using the same decoder as the inference user uses [see "def save_image_act_pair"].
            if self.gan_AE_type == "custom":
                decoder2 = architectures.custom_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                    output_dim=32, activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "custom_bn":
                decoder2 = architectures.custom_AE_bn(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                        output_dim=32, activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "complex":
                decoder2 = architectures.complex_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                    output_dim=32, activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "complex_plus":
                decoder2 = architectures.complex_plus_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                            output_dim=32, activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "complex_res":
                decoder2 = architectures.complex_res_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                        output_dim=32, activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "complex_resplus":
                decoder2 = architectures.complex_resplus_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                            output_dim=32, activation=self.gan_AE_activation).cuda()
            elif "complex_resplusN" in self.gan_AE_type:
                try:
                    N = int(self.gan_AE_type.split("complex_resplusN")[1])
                except:
                    print("auto extract N from complex_resplusN failed, set N to default 2")
                    N = 2
                decoder2 = architectures.complex_resplusN_AE(N = N, input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                            output_dim=32, activation=self.gan_AE_activation).cuda()
            elif "complex_normplusN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("complex_normplusN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from complex_normplusN failed, set N to default 2")
                    N = 0
                    internal_C = 64
                decoder2 = architectures.complex_normplusN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                            input_dim=input_dim, output_dim=32,
                                                            activation=self.gan_AE_activation).cuda()

            elif "conv_normN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("conv_normN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from conv_normN failed, set N to default 2")
                    N = 0
                    internal_C = 64
                decoder2 = architectures.conv_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                            input_dim=input_dim, output_dim=32,
                                                            activation=self.gan_AE_activation).cuda()
            
            elif "res_normN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("res_normN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from res_normN failed, set N to default 2")
                    N = 0
                    internal_C = 64
                decoder2 = architectures.res_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                            input_dim=input_dim, output_dim=32,
                                                            activation=self.gan_AE_activation).cuda()

            elif "TB_normplusN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("TB_normplusN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from TB_normplusN failed, set N to default 2")
                    N = 0
                    internal_C = 64
                decoder2 = architectures.TB_normplusN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                            input_dim=input_dim, output_dim=32,
                                                            activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "simple":
                decoder2 = architectures.simple_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                    output_dim=32, activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "simple_bn":
                decoder2 = architectures.simple_AE_bn(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                        output_dim=32, activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "simplest":
                decoder2 = architectures.simplest_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                        output_dim=32, activation=self.gan_AE_activation).cuda()
            else:
                raise ("No such GAN AE type.")
            # optimizer2 = torch.optim.Adam(decoder2.parameters(), lr=1e-3)
            optimizer2 = torch.optim.Adam(decoder2.parameters(), lr=1e-3)
            self.attack(attack_num_epochs, decoder2, optimizer2, trainloader, testloader, logger, path_dict,
                        attack_batchsize, pretrained_decoder=self.local_AE_list[collude_client], noise_aware=noise_aware)
            decoder = decoder2  # use decoder2 for testing
        else:
            # Perform Input Extraction Attack
            self.attack(attack_num_epochs, decoder, optimizer, trainloader, testloader, logger, path_dict,
                        attack_batchsize, noise_aware=noise_aware, loss_type=loss_type)

        
        # malicious_option = True if "local_plus_sampler" in args.MA_fashion else False
        mse_score, ssim_score, psnr_score = self.test_attack(attack_num_epochs, decoder, sp_testloader, logger,
                                                                path_dict, attack_batchsize,
                                                                num_classes=self.num_class)

        # Clear content of image_data_dir/tensor_data_dir
        if os.path.isdir(image_data_dir):
            rmtree(image_data_dir)
        if os.path.isdir(tensor_data_dir):
            rmtree(tensor_data_dir)
        return mse_score, ssim_score, psnr_score
    elif attack_option == "MIA_mf":  # Stands for Model-free MIA, does not need a AE model, optimize each fake image instead.

        lambda_TV = 0.0
        lambda_l2 = 0.0
        num_step = attack_num_epochs * 60

        sp_testloader = apply_transform_test(1, self.save_dir + "/save_activation_client_{}_epoch_{}".format(
            target_client, self.n_epochs), self.save_dir + "/save_activation_client_{}_epoch_{}".format(target_client,
                                                                                                        self.n_epochs))
        criterion = nn.MSELoss().cuda()
        ssim_loss = pytorch_ssim.SSIM()
        all_test_losses = AverageMeter()
        ssim_test_losses = AverageMeter()
        psnr_test_losses = AverageMeter()
        fresh_option = True
        for num, data in enumerate(sp_testloader, 1):
            # img, ir, _ = data
            img, ir, _ = data

            # optimize a fake_image to (1) have similar ir, (2) have small total variance, (3) have small l2
            img = img.cuda()
            if not fresh_option:
                ir = ir.cuda()
            self.model.local_list[collude_client].eval()
            self.model.local_list[target_client].eval()

            fake_image = torch.zeros(img.size(), requires_grad=True, device="cuda")
            optimizer = torch.optim.Adam(params=[fake_image], lr=8e-1, amsgrad=True, eps=1e-3)
            # optimizer = torch.optim.Adam(params = [fake_image], lr = 1e-2, amsgrad=True, eps=1e-3)
            for step in range(1, num_step + 1):
                optimizer.zero_grad()

                fake_ir = self.model.local_list[collude_client](fake_image)  # Simulate Original

                if fresh_option:
                    ir = self.model.local_list[target_client](img)  # Getting fresh ir from target local model

                featureLoss = criterion(fake_ir, ir)

                TVLoss = TV(fake_image)
                normLoss = l2loss(fake_image)

                totalLoss = featureLoss + lambda_TV * TVLoss + lambda_l2 * normLoss

                totalLoss.backward()

                optimizer.step()
                # if step % 100 == 0:
                if step == 0 or step == num_step:
                    logger.debug("Iter {} Feature loss: {} TVLoss: {} l2Loss: {}".format(step,
                                                                                            featureLoss.cpu().detach().numpy(),
                                                                                            TVLoss.cpu().detach().numpy(),
                                                                                            normLoss.cpu().detach().numpy()))
            imgGen = fake_image.clone()
            imgOrig = img.clone()

            mse_loss = criterion(imgGen, imgOrig)
            ssim_loss_val = ssim_loss(imgGen, imgOrig)
            psnr_loss_val = get_PSNR(imgOrig, imgGen)
            all_test_losses.update(mse_loss.item(), ir.size(0))
            ssim_test_losses.update(ssim_loss_val.item(), ir.size(0))
            psnr_test_losses.update(psnr_loss_val.item(), ir.size(0))
            if not os.path.isdir(test_output_path + "/{}".format(attack_num_epochs)):
                os.mkdir(test_output_path + "/{}".format(attack_num_epochs))
            if DENORMALIZE_OPTION:
                imgGen = denormalize(imgGen, self.dataset)
            torchvision.utils.save_image(imgGen, test_output_path + '/{}/out_{}.jpg'.format(attack_num_epochs,
                                                                                                num * attack_batchsize + attack_batchsize))
            if DENORMALIZE_OPTION:
                imgOrig = denormalize(imgOrig, self.dataset)
            torchvision.utils.save_image(imgOrig, test_output_path + '/{}/inp_{}.jpg'.format(attack_num_epochs,
                                                                                                num * attack_batchsize + attack_batchsize))
            # imgGen = deprocess(imgGen, self.num_class)
            # imgOrig = deprocess(imgOrig, self.num_class)
            # torchvision.utils.save_image(imgGen, test_output_path + '/{}/dp_out_{}.jpg'.format(attack_num_epochs, num*attack_batchsize + attack_batchsize))
            # torchvision.utils.save_image(imgOrig, test_output_path + '/{}/dp_inp_{}.jpg'.format(attack_num_epochs, num*attack_batchsize + attack_batchsize))
        logger.debug("MSE Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(
            all_test_losses.avg))
        logger.debug("SSIM Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(
            ssim_test_losses.avg))
        logger.debug("PSNR Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(
            psnr_test_losses.avg))
        return all_test_losses.avg, ssim_test_losses.avg, psnr_test_losses.avg

def attack(self, num_epochs, decoder, optimizer, trainloader, testloader, logger, path_dict, batch_size,
            loss_type="MSE", pretrained_decoder=None, noise_aware=False):
    round_ = 0
    min_val_loss = 999.
    max_val_loss = 0.
    train_output_freq = 10
    # test_output_freq = 50
    train_losses = AverageMeter()
    val_losses = AverageMeter()

    # Optimize based on MSE distance
    if loss_type == "MSE":
        criterion = nn.MSELoss()
    elif loss_type == "SSIM":
        criterion = pytorch_ssim.SSIM()
    elif loss_type == "PSNR":
        criterion = None
    else:
        raise ("No such loss in self.attack")
    device = next(decoder.parameters()).device
    decoder.train()
    for epoch in range(round_ * num_epochs, (round_ + 1) * num_epochs):
        for num, data in enumerate(trainloader, 1):
            img, ir = data
            img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
            img, ir = Variable(img).to(device), Variable(ir).to(device)
            # print(img)
            # Use local DP for training the AE.
            if self.local_DP and noise_aware:
                with torch.no_grad():
                    if "laplace" in self.regularization_option:
                        ir += torch.from_numpy(
                            np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=ir.size())).cuda()
                    else:  # apply gaussian noise
                        delta = 10e-5
                        sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                        ir += sigma * torch.randn_like(ir).cuda()
            if self.dropout_defense and noise_aware:
                ir = dropout_defense(ir, self.dropout_ratio)
            if self.topkprune and noise_aware:
                ir = prune_defense(ir, self.topkprune_ratio)
            if pretrained_decoder is not None and "gan_adv_noise" in self.regularization_option and noise_aware:
                epsilon = self.alpha2
                
                pretrained_decoder.eval()
                fake_act = ir.clone()
                grad = torch.zeros_like(ir).cuda()
                fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
                x_recon = pretrained_decoder(fake_act)
                if self.gan_loss_type == "SSIM":
                    ssim_loss = pytorch_ssim.SSIM()
                    loss = ssim_loss(x_recon, img)
                    loss.backward()
                    grad -= torch.sign(fake_act.grad)
                else:
                    mse_loss = nn.MSELoss()
                    loss = mse_loss(x_recon, img)
                    loss.backward()
                    grad += torch.sign(fake_act.grad)
                # ir = ir + grad.detach() * epsilon
                ir = ir - grad.detach() * epsilon
            # print(ir.size())
            output = decoder(ir)

            if loss_type == "MSE":
                reconstruction_loss = criterion(output, img)
            elif loss_type == "SSIM":
                reconstruction_loss = -criterion(output, img)
            elif loss_type == "PSNR":
                reconstruction_loss = -1 / 10 * get_PSNR(img, output)
            else:
                raise ("No such loss in self.attack")
            train_loss = reconstruction_loss

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_losses.update(train_loss.item(), ir.size(0))

        if (epoch + 1) % train_output_freq == 0:
            save_images(img, output, epoch, path_dict["train_output_path"], offset=0, batch_size=batch_size)

        for num, data in enumerate(testloader, 1):
            img, ir = data

            img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
            img, ir = Variable(img).to(device), Variable(ir).to(device)

            output = decoder(ir)

            reconstruction_loss = criterion(output, img)
            val_loss = reconstruction_loss

            if loss_type == "MSE" and val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(decoder.state_dict(), path_dict["model_path"])
            elif loss_type == "SSIM" and val_loss > max_val_loss:
                max_val_loss = val_loss
                torch.save(decoder.state_dict(), path_dict["model_path"])
            elif loss_type == "PSNR" and val_loss > max_val_loss:
                max_val_loss = val_loss
                torch.save(decoder.state_dict(), path_dict["model_path"])
            val_losses.update(val_loss.item(), ir.size(0))

        # torch.save(decoder.state_dict(), path_dict["model_path"])
        logger.debug(
            "epoch [{}/{}], train_loss {train_losses.val:.4f} ({train_losses.avg:.4f}), val_loss {val_losses.val:.4f} ({val_losses.avg:.4f})".format(
                epoch + 1,
                num_epochs, train_losses=train_losses, val_losses=val_losses))
    if loss_type == "MSE":
        logger.debug("Best Validation Loss is {}".format(min_val_loss))
    elif loss_type == "SSIM":
        logger.debug("Best Validation Loss is {}".format(max_val_loss))
    elif loss_type == "PSNR":
        logger.debug("Best Validation Loss is {}".format(max_val_loss))

def test_attack(self, num_epochs, decoder, sp_testloader, logger, path_dict, batch_size, num_classes=10,
                select_label=0):
    device = next(decoder.parameters()).device
    # print("Load the best Decoder Model...")
    new_state_dict = torch.load(path_dict["model_path"])
    decoder.load_state_dict(new_state_dict)
    decoder.eval()
    # test_losses = []
    all_test_losses = AverageMeter()
    ssim_test_losses = AverageMeter()
    psnr_test_losses = AverageMeter()
    ssim_loss = pytorch_ssim.SSIM()

    criterion = nn.MSELoss()

    for num, data in enumerate(sp_testloader, 1):
        img, ir, label = data

        img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
        img, ir = Variable(img).to(device), Variable(ir).to(device)
        output_imgs = decoder(ir)
        reconstruction_loss = criterion(output_imgs, img)
        ssim_loss_val = ssim_loss(output_imgs, img)
        psnr_loss_val = get_PSNR(img, output_imgs)
        all_test_losses.update(reconstruction_loss.item(), ir.size(0))
        ssim_test_losses.update(ssim_loss_val.item(), ir.size(0))
        psnr_test_losses.update(psnr_loss_val.item(), ir.size(0))
        save_images(img, output_imgs, num_epochs, path_dict["test_output_path"], offset=num, batch_size=batch_size)

    logger.debug(
        "MSE Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(all_test_losses.avg))
    logger.debug(
        "SSIM Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(ssim_test_losses.avg))
    logger.debug(
        "PSNR Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(psnr_test_losses.avg))
    return all_test_losses.avg, ssim_test_losses.avg, psnr_test_losses.avg

def save_activation_bhtsne(self, save_activation, labels, batch_size, msg1, msg2):
    """
        Run one train epoch
    """

    save_activation = save_activation.float()
    save_activation = save_activation.detach().cpu().numpy()
    save_activation = save_activation.reshape(batch_size, -1)
    f=open(os.path.join(self.save_dir, "{}_{}_act.txt".format(msg1, msg2)),'a')
    np.savetxt(f, save_activation, fmt='%.2f')
    f.close()

    target = labels.cpu().numpy()
    target = target.reshape(batch_size, -1)
    f=open(os.path.join(self.save_dir, "{}_{}_target.txt".format(msg1, msg2)),'a')
    np.savetxt(f, target, fmt='%d')
    f.close()

#Generate test set for MIA decoder
def save_image_act_pair(self, input, target, client_id, epoch, clean_option=False, attack_from_later_layer=-1, attack_option = "MIA"):
    """
        Run one train epoch
    """
    path_dir = os.path.join(self.save_dir, 'save_activation_client_{}_epoch_{}'.format(client_id, epoch))
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)
    else:
        rmtree(path_dir)
        os.makedirs(path_dir)
    input = input.cuda()

    for j in range(input.size(0)):
        img = input[None, j, :, :, :]
        label = target[None, j]
        with torch.no_grad():
            if client_id == 0:
                self.f.eval()
                save_activation = self.f(img)
            elif client_id == 1:
                self.c.eval()
                save_activation = self.c(img)
            elif client_id > 1:
                self.model.local_list[client_id].eval()
                save_activation = self.model.local_list[client_id](img)
            if self.confidence_score:
                self.model.cloud.eval()
                save_activation = self.model.cloud(save_activation)
                if self.arch == "resnet18" or self.arch == "resnet34" or "mobilenetv2" in self.arch:
                    save_activation = F.avg_pool2d(save_activation, 4)
                    save_activation = save_activation.view(save_activation.size(0), -1)
                    save_activation = self.classifier(save_activation)
                elif self.arch == "resnet20" or self.arch == "resnet32":
                    save_activation = F.avg_pool2d(save_activation, 8)
                    save_activation = save_activation.view(save_activation.size(0), -1)
                    save_activation = self.classifier(save_activation)
                else:
                    save_activation = save_activation.view(save_activation.size(0), -1)
                    save_activation = self.classifier(save_activation)
        

        if attack_from_later_layer > -1 and (not self.confidence_score):
            self.model.cloud.eval()

            activation_3 = {}

            def get_activation_3(name):
                def hook(model, input, output):
                    activation_3[name] = output.detach()

                return hook

            with torch.no_grad():
                activation_3 = {}
                count = 0
                for name, m in self.model.cloud.named_modules():
                    if attack_from_later_layer == count:
                        m.register_forward_hook(get_activation_3("ACT-{}".format(name)))
                        valid_key = "ACT-{}".format(name)
                        break
                    count += 1
                output = self.model.cloud(save_activation)
            try:
                save_activation = activation_3[valid_key]
            except:
                print("cannot attack from later layer, server-side model is empty or does not have enough layers")
        
        if self.local_DP and not clean_option:  # local DP or additive noise
            if "laplace" in self.regularization_option:
                save_activation += torch.from_numpy(
                    np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=save_activation.size())).cuda()
                # the addtive work uses scale in (0.1 0.5 1.0) -> (1 2 10) regularization_strength (self.dp_epsilon)
            else:  # apply gaussian noise
                delta = 10e-5
                sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                save_activation += sigma * torch.randn_like(save_activation).cuda()
        if self.dropout_defense and not clean_option:  # activation dropout defense
            save_activation = dropout_defense(save_activation, self.dropout_ratio)
        if self.topkprune and not clean_option:
            save_activation = prune_defense(save_activation, self.topkprune_ratio)
        if DENORMALIZE_OPTION:
            img = denormalize(img, self.dataset)
            
        if self.gan_noise and not clean_option:
            epsilon = self.alpha2
            self.local_AE_list[client_id].eval()
            fake_act = save_activation.clone()
            grad = torch.zeros_like(save_activation).cuda()
            fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
            x_recon = self.local_AE_list[client_id](fake_act)
            
            if self.gan_loss_type == "SSIM":
                ssim_loss = pytorch_ssim.SSIM()
                loss = ssim_loss(x_recon, img)
                loss.backward()
                grad -= torch.sign(fake_act.grad)
            elif self.gan_loss_type == "MSE":
                mse_loss = torch.nn.MSELoss()
                loss = mse_loss(x_recon, img)
                loss.backward()
                grad += torch.sign(fake_act.grad)  

            save_activation = save_activation - grad.detach() * epsilon
        if "truncate" in attack_option:
            save_activation = prune_top_n_percent_left(save_activation)
        
        save_activation = save_activation.float()
        
        save_image(img, os.path.join(path_dir, "{}.jpg".format(j)))
        torch.save(save_activation.cpu(), os.path.join(path_dir, "{}.pt".format(j)))
        torch.save(label.cpu(), os.path.join(path_dir, "{}.label".format(j)))

# if __name__ == '__main__':
#     a = [torch.ones([128, 8]), torch.ones([128, 8]), torch.ones([128, 8]), torch.ones([128, 8])]
#     max_grad, max_idx = label_deduction(a)
#     print(max_idx)
#     print(max_grad.size())
#     print(max_idx.size())