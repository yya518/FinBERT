import os
import copy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from bertModel import BertClassification, dense_opt
from datasets import text_dataset, financialPhraseBankDataset
import argparse
from sklearn.metrics import f1_score

def train_model(model, model_type, path, criterion, optimizer, scheduler, device, num_epochs=100, early_stopping = 7):
    model.to(device)
    log_file =  os.path.join(path, "{}_log.txt".format(model_type))
    model_path = os.path.join(path, "{}.pth".format(model_type))
    wo= open(log_file, 'w')
    
    since = time.time()
    print('starting')
    wo.write('starting \n')
    best_loss = 100
    best_accuracy = 0 
    best_f1 = 0
    early_stopping_count = 0
    
    for epoch in range(num_epochs):
        if (early_stopping_count >= early_stopping):
            break 
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        wo.write('Epoch {}/{} \n'.format(epoch, num_epochs - 1))
        print('-' * 10)
        wo.write('-' * 10 + "\n")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
                early_stopping_count +=1

            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            sentiment_corrects = 0
            actual = torch.tensor([]).long().to(device)
            pred = torch.tensor([]).long().to(device)

            # Iterate over data.
            for inputs, sentiment in dataloaders_dict[phase]:
           
                input_ids = inputs["input_ids"].to(device)
                token_type_ids = inputs["token_type_ids"].to(device)
                attention_mask  =  inputs["attention_mask"].to(device)
                sentiment = sentiment.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(input_ids, token_type_ids, attention_mask)
                    outputs = F.softmax(outputs,dim=1)
                    loss = criterion(outputs, torch.max(sentiment.float(), 1)[1])
        
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * input_ids.size(0)
                sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(sentiment, 1)[1])
                actual = torch.cat([actual, torch.max(outputs, 1)[1]], dim=0)
                pred= torch.cat([pred, torch.max(sentiment, 1)[1]], dim=0)
            epoch_loss = running_loss / dataset_sizes[phase]

            sentiment_acc = sentiment_corrects.double() / dataset_sizes[phase]
            assert(len(actual) == len(pred))
            assert(len(actual) == dataset_sizes[phase])
            f1 = f1_score(actual.cpu().numpy(), pred.cpu().numpy(), average='weighted')
            
            print('{} total loss (avg): {:.4f} '.format(phase,epoch_loss ))
            wo.write('{} total loss: {:.4f} \n'.format(phase,epoch_loss ))
            print('{} sentiment_acc: {:.4f}'.format(phase, sentiment_acc))
            wo.write('{} sentiment_acc: {:.4f} \n'.format(phase, sentiment_acc))
            print('{} f1-score: {:.4f}'.format(phase, f1))
            wo.write('{} f1-score:: {:.4f} \n'.format(phase, f1))

            if phase == 'val' and epoch_loss < best_loss:
                print('saving with loss of {}'.format(epoch_loss),
                      'improved over previous {}'.format(best_loss))
                wo.write('saving with loss of {} \n'.format(epoch_loss))
                wo.write('improved over previous {} \n'.format(best_loss))
                wo.write("\n")
                best_loss = epoch_loss
                best_accuracy = sentiment_acc
                best_f1 = f1
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_path)
                early_stopping_count = 0
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    wo.write('Training complete in {:.0f}m {:.0f}s \n'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:8f}'.format(float(best_accuracy)))
    wo.write('Best val Acc: {:8f} \n'.format(float(best_accuracy)))
    print('Best val f1: {:8f}'.format(float(best_f1)))
    wo.write('Best val f1: {:8f} \n'.format(float(best_f1)))
    wo.close()
    
    # load best model weights
    model.load_state_dict(best_model_wts)

    return best_accuracy, best_f1

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda_device', type=str, default="cuda:0")
    parser.add_argument('--vocab', type= str, default = "base-cased")
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--vocab_path', type=str)
    parser.add_argument('--data_dir', type=str) 
    parser.add_argument('--weight_path', type=str) 
    args = parser.parse_args()
    
    num_labels = 3
    vocab_to_model_dict = { "base-cased": "FinBert_BaseVocab_Cased", 
                            "base-uncased": "FinBert_BaseVocab_Uncased",
                            "finance-cased": "FinBert_FinVocab_Cased",
                            "finance-uncased": "FinBert_FinVocab_Uncased"}
    
    model_type = vocab_to_model_dict[args.vocab]

    list_of_train_splits = financialPhraseBankDataset(args.data_dir)
    
    X_train, X_test, y_train, y_test =  list_of_train_splits
    
    train_lists = [X_train, y_train]
    test_lists = [X_test, y_test]

    training_dataset = text_dataset(x_y_list = train_lists, vocab= args.vocab, vocab_path = args.vocab_path)
    test_dataset = text_dataset(x_y_list = test_lists , vocab = args.vocab, vocab_path = args.vocab_path )

    dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0), 'val':torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)}

    dataset_sizes = {'train':len(train_lists[0]),
                'val':len(test_lists[0])}

    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")

    model = BertClassification(weight_path=args.weight_path, num_labels=num_labels, vocab=args.vocab)
    Dense_opt = dense_opt(model)
    optim = Dense_opt.get_optim()

    criterion = nn.CrossEntropyLoss()
    
    exp_lr_scheduler = lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)
    train_model(model, model_type, args.output_path, criterion, optim, exp_lr_scheduler, device)