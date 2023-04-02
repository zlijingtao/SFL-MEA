import copy
class virtual_module_class():
    def __init__(self, object_list, object_name_list, number_layer_to_freeze) -> None:
        self.object_list = object_list
        self.object_name_list = object_name_list
        self.number_layer_to_freeze = number_layer_to_freeze

    def parameters(self): # only get the trainable parameters
        tranable_param_list = []
        count = 0
        # count = 1: embedding, count = 2: layer[0]
        if self.number_layer_to_freeze == -1:
            for item in self.object_list:
                tranable_param_list += list(item.parameters())
        else: # only submit active layer
            for item in self.object_list:
                count += 1
                if count > 1 + self.number_layer_to_freeze:
                    tranable_param_list += list(item.parameters())
        return tranable_param_list
    
    def load_state_dict(self, state_dict, strict = True):
        for name, module in zip(self.object_name_list, self.object_list):
            module.load_state_dict(state_dict[name], strict = strict)
    
    def state_dict(self):
        state_dict = {}
        for name, module in zip(self.object_name_list, self.object_list):
            state_dict[name] = module.state_dict()
        return state_dict
    
    def apply(self, apply_functions):
        for item in self.object_list:
            item.apply(apply_functions)
    
    def cuda(self):
        for item in self.object_list:
            item.cuda()
    def cpu(self):
        for item in self.object_list:
            item.cpu()
    def train(self):
        for item in self.object_list:
            item.train()
    def eval(self):
        for item in self.object_list:
            item.eval()
    def named_parameters(self):
        named_parameter_list = []
        for item in self.object_list:
            for name, param in item.named_parameters():
                named_parameter_list.append((name, param))
        return named_parameter_list
    
    def modules(self):
        return self.object_list


class vit_split_model_wrapper():
    def __init__(self, model, cutlayer, num_client, number_layer_to_freeze) -> None:
        self.num_client = num_client
        self.cutlayer = cutlayer
        self.model = model
        self.trainable_object_list = [self.model.vit.embeddings, *self.model.vit.encoder.layer, self.model.vit.layernorm, self.model.classifier]
        self.trainable_object_name_list = ["vit.embeddings", "vit.encoder.layer.0", "vit.encoder.layer.1", "vit.encoder.layer.2", "vit.encoder.layer.2"
                                           , "vit.encoder.layer.4", "vit.encoder.layer.5", "vit.encoder.layer.6", "vit.encoder.layer.7"
                                           , "vit.encoder.layer.8", "vit.encoder.layer.9", "vit.encoder.layer.10", "vit.encoder.layer.11", "vit.layernorm", "classifier"]
        
        if cutlayer < 0 or cutlayer > 12:
            raise("invalid cutlayer for vit")
        cloud_object_list = self.trainable_object_list[cutlayer + 1:]
        cloud_object_name_list = self.trainable_object_name_list[cutlayer + 1:]
        local_object_name_list = self.trainable_object_name_list[:cutlayer + 1]
        self.cloud = virtual_module_class(cloud_object_list, cloud_object_name_list, -1)
        
        self.freeze_front_layer(number_layer_to_freeze)
        self.eval()

        self.model_list = [self.model]
        self.local_list = [virtual_module_class(self.trainable_object_list[:cutlayer + 1], local_object_name_list, number_layer_to_freeze)]
        self.local = self.local_list[0]
        for i in range(1, num_client):
            self.model_list.append(copy.deepcopy(self.model)) # create entire copy of the model

            # share the cloud part
            for layer in range(cutlayer, 12):
                self.model_list[i].vit.encoder.layer[layer] = self.model.vit.encoder.layer[layer]
            if cutlayer <= 11:
                self.model_list[i].vit.layernorm = self.model.vit.layernorm
                self.model_list[i].classifier = self.model.classifier

            # pack these object into a list, get their parameters
            temp_object_list = [self.model_list[i].vit.embeddings, *self.model_list[i].vit.encoder.layer, self.model_list[i].vit.layernorm, self.model_list[i].classifier]
            
            self.local_list.append(virtual_module_class(temp_object_list[:(cutlayer + 1)], local_object_name_list, number_layer_to_freeze))
            print(f"Shared client-side model: { self.model_list[i].vit.embeddings is self.model_list[0].vit.embeddings}")
            print(f"Shared server-side model: { self.model_list[i].classifier is self.model_list[0].classifier}")
            print(f"Shared server-side model: { self.model_list[i].vit.encoder.layer[11] is self.model_list[0].vit.encoder.layer[11]}")

    def switch_model(self, client_id = 0):
        print(f"switch to client-{client_id}")
        if client_id > self.num_client - 1:
            raise("invalid client_id!")
        # self.model.load_state_dict({**self.local_dict_list[client_id], **self.cloud_dict})
        self.model = self.model_list[client_id]
    
    def freeze_front_layer(self, number_layer_to_freeze = 2):

        if number_layer_to_freeze < 0 or number_layer_to_freeze > self.cutlayer:
            raise("invalid cutlayer for vit")
        number_layer_to_freeze = number_layer_to_freeze + 1 # always freeze embedding layer
        
        count = 0
        for module in self.trainable_object_list:
            if count < number_layer_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True
            count += 1

    def get_num_of_local_layer(self):
        return self.cutlayer

    def get_num_of_cloud_layer(self):
        return 14 - self.cutlayer

    def merge_classifier_cloud(self):
        pass
    
    def resplit(self, number_layer_at_cloud):
        number_layer_at_cloud = int(number_layer_at_cloud) # make sure it is an integer
        if number_layer_at_cloud == 14 - self.cutlayer:
            print("cutlayer is already satisfied, ignore resplit")
        else:
            # raise("cannot support resplit for now, please reinitialize with a different cutting layer")

            for i in range(self.num_client):
                temp_object_list = [self.model_list[i].vit.embeddings, *self.model_list[i].vit.encoder.layer, self.model_list[i].vit.layernorm, self.model_list[i].classifier]
                
                self.local_list[i].object_list = temp_object_list[:15-number_layer_at_cloud]
                self.local_list[i].object_name_list = self.trainable_object_name_list[:15-number_layer_at_cloud]
                self.local_list[i].number_layer_to_freeze = 14 - number_layer_at_cloud # corresponding cut-layer

                self.cloud.object_list = temp_object_list[15-number_layer_at_cloud:]
                self.cloud.object_name_list = self.trainable_object_name_list[15-number_layer_at_cloud:]
                self.cloud.number_layer_to_freeze = -1
                # pack these object into a list, get their parameters

    def __call__(self, x):
        return self.model(x).logits

    def eval(self): # set the entire model to eval mode
        self.model.eval()
    
    def train(self): # set the trainable part to train mode
        for module in self.trainable_object_list:
            module.train()

    def cuda(self):
        self.model.cuda()  
    
    def cpu(self):
        self.model.cpu()