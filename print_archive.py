#%%
from data.parallelzipfilebetter import ParallelZipFile as ZipFile
import cloudpickle

class ZipStreamReader:
    def __init__(self, dir:str) -> None:
        self.zip = ZipFile(file=dir, mode='r')
        self.files = sorted(self.zip.namelist())
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        x = self.zip.read(self.files[idx])
        # loaded = np.load(BytesIO(x), allow_pickle=True)
        x,y = cloudpickle.loads(x)
        return x, y

#df = ZipStreamReader('cp_dataset_train_all.zip')

df = ZipStreamReader('cp_dataset_train_w.zip')
#df = ZipStreamReader('fixed.zip')
print(len(df))
it = iter(df)
for i in range(200):
    next(it)
for i in range(5):
    next(it)
    next(it)
    next(it)
    x,y = next(it)

    print(len(df))

    from data.dataloaders import ProgramEncoder
    print(x.keys())

    prog_enc = ProgramEncoder(15)
    print(prog_enc.decode_pred(y))
#df.zip.close()