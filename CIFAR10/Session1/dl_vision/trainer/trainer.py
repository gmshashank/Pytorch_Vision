import torch
from typing import List, Tuple
from base.base_trainer import BaseTrainer
from utils.logger import setup_logger
from tqdm.notebook import tqdm

logger = setup_logger(__name__)


class Trainer(BaseTrainer):
    def __init__(
        self, model, optimizer, loss, config, device, train_loader, test_loader
    ):
        super().__init__(model, optimizer, loss, config, device)

        self.train_loader = train_loader
        self.test_loader = test_loader

    def _train_epoch(self, epoch: int) -> List[Tuple]:
        loss_coll = []
        accuracy_coll = []

        self.model.train()

        train_loss = 0
        total = 0
        processed = 0
        correct = 0

        pbar = tqdm(self.train_loader)

        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            processed += len(data)

            # pbar.set_description(desc=f'epoch={epoch+batch_idx/len(pbar):.2f} | loss={train_loss/(batch_idx+1):.10f} | accuracy={100.*correct/total:.2f} {correct}/{total} | batch_id={batch_idx}')
            pbar.set_description(
                desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy: {100*correct/processed:0.2f}% "
            )
            accuracy_coll.append(100 * correct / processed)
            loss_coll.append(loss.data.cpu().numpy().item())

        logger.info(
            f"Train Set: Epoch {epoch}, Average Loss {train_loss/len(self.train_loader):.5f}, Accuracy: {100*correct/total}%  ({correct/total})"
        )
        print("Training Done.")

        return (loss_coll, accuracy_coll)

    def _test_epoch(self, epoch: int) -> List[Tuple]:
        loss_coll = []
        accuracy_coll = []

        self.model.eval()

        test_loss = 0
        total = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        logger.info(
            f"Test Set: Epoch {epoch}, Average Loss {test_loss/len(self.test_loader):.5f}, Accuracy: {100*correct/total}%  ({correct/total})"
        )

        loss_coll.append(test_loss / len(self.test_loader))
        accuracy_coll.append(100 * correct / total)

        return (loss_coll, accuracy_coll, test_loss)
