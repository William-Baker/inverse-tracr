from data.dataloader_streams import StreamReader
pth = '.data/iTracr_dataset/'
r = StreamReader(pth)

it = iter(r)
x,y = next(it)