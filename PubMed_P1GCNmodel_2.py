import torch
import ptens
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from Transforms import ToPtens_Batch
dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())
data = dataset[0]  
transform_nodes = RandomNodeSplit(split = 'train_rest', 
                                  num_val = 3154,
                                  num_test = 3943)
data = transform_nodes(data)
on_learn_transform = ToPtens_Batch()
data = on_learn_transform(data)

class P1GCN(torch.nn.Module):
    def __init__(self, hidden_channels, reduction_type):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = ptens.modules.ConvolutionalLayer_1P(dataset.num_features, hidden_channels, reduction_type)
        self.conv2 = ptens.modules.ConvolutionalLayer_1P(hidden_channels, dataset.num_classes, reduction_type)
        self.dropout = ptens.modules.Dropout(prob=0.5,device = None)

    def forward(self, x, edge_index):
        x = ptens.linmaps1(x, False)
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = ptens.linmaps0(x, False).torch().sum(axis = 1)
        x = F.log_softmax(x, dim=1)
        x = ptens.ptensors0.from_matrix(x)
        return x

def train():
      model.train()
      optimizer.zero_grad()
      data_x = ptens.ptensors0.from_matrix(data.x)
      out = model(data_x,data.G).torch()
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward(retain_graph=True) 
      optimizer.step() 
      return loss
def test():
      model.eval()
      data_x = ptens.ptensors0.from_matrix(data.x)
      out = model(data_x,data.G).torch()
      pred = out.argmax(dim=1)  
      train_correct = pred[data.train_mask] == data.y[data.train_mask]  
      train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) 
      return train_acc, test_acc

    
model = P1GCN(hidden_channels = 32, reduction_type = "mean") 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=8e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
'''
Epoch: 001, Loss: 4.4802
Train Accuracy: 0.4397781299524564 . Test Accuracy: 0.45041846309916306 .
Epoch: 002, Loss: 3.3485
Train Accuracy: 0.5012678288431062 . Test Accuracy: 0.5135683489728633 .
Epoch: 003, Loss: 7.4375
Train Accuracy: 0.49833597464342316 . Test Accuracy: 0.48871417702257164 .
Epoch: 004, Loss: 3.8010
Train Accuracy: 0.5072107765451664 . Test Accuracy: 0.5191478569617043 .
Epoch: 005, Loss: 3.3563
Train Accuracy: 0.5721870047543581 . Test Accuracy: 0.5812832868374335 .
Epoch: 006, Loss: 6.9718
Train Accuracy: 0.6160063391442155 . Test Accuracy: 0.6193253867613492 .
Epoch: 007, Loss: 3.8861
Train Accuracy: 0.44136291600633915 . Test Accuracy: 0.429368501141263 .
Epoch: 008, Loss: 6.6507
Train Accuracy: 0.487797147385103 . Test Accuracy: 0.49378645701242707 .
Epoch: 009, Loss: 3.1271
Train Accuracy: 0.5773375594294771 . Test Accuracy: 0.5914278468171443 .
Epoch: 010, Loss: 5.1038
Train Accuracy: 0.43502377179080826 . Test Accuracy: 0.43824499112351 .
Epoch: 011, Loss: 4.2144
Train Accuracy: 0.5317749603803487 . Test Accuracy: 0.5356327669287345 .
Epoch: 012, Loss: 2.3710
Train Accuracy: 0.5305071315372425 . Test Accuracy: 0.5445092569109815 .
Epoch: 013, Loss: 3.8273
Train Accuracy: 0.6719492868462758 . Test Accuracy: 0.6715698706568602 .
Epoch: 014, Loss: 2.9633
Train Accuracy: 0.6383518225039619 . Test Accuracy: 0.6479837687040325 .
Epoch: 015, Loss: 3.9976
Train Accuracy: 0.6455625990491284 . Test Accuracy: 0.6472229267055541 .
Epoch: 016, Loss: 4.1877
Train Accuracy: 0.6403328050713154 . Test Accuracy: 0.6434187167131625 .
Epoch: 017, Loss: 3.8102
Train Accuracy: 0.6709984152139461 . Test Accuracy: 0.6766421506467157 .
Epoch: 018, Loss: 3.9177
Train Accuracy: 0.6696513470681458 . Test Accuracy: 0.6682728886634542 .
Epoch: 019, Loss: 2.1777
Train Accuracy: 0.6187004754358162 . Test Accuracy: 0.6160284047679432 .
Epoch: 020, Loss: 4.1993
Train Accuracy: 0.7328050713153724 . Test Accuracy: 0.7380167385239665 .
Epoch: 021, Loss: 2.8118
Train Accuracy: 0.7134706814580032 . Test Accuracy: 0.7200101445599797 .
Epoch: 022, Loss: 1.8342
Train Accuracy: 0.6896988906497623 . Test Accuracy: 0.6959168146081663 .
Epoch: 023, Loss: 1.9923
Train Accuracy: 0.6855784469096672 . Test Accuracy: 0.6913517626172965 .
Epoch: 024, Loss: 2.2327
Train Accuracy: 0.6882725832012678 . Test Accuracy: 0.7070758305858483 .
Epoch: 025, Loss: 2.1169
Train Accuracy: 0.6812202852614897 . Test Accuracy: 0.6860258686279482 .
Epoch: 026, Loss: 2.5675
Train Accuracy: 0.6166402535657686 . Test Accuracy: 0.626172964747654 .
Epoch: 027, Loss: 2.4839
Train Accuracy: 0.7121236133122029 . Test Accuracy: 0.7169667765660664 .
Epoch: 028, Loss: 2.2097
Train Accuracy: 0.7334389857369256 . Test Accuracy: 0.7364950545270099 .
Epoch: 029, Loss: 2.7112
Train Accuracy: 0.6450871632329636 . Test Accuracy: 0.6555921886888156 .
Epoch: 030, Loss: 1.5693
Train Accuracy: 0.5403328050713154 . Test Accuracy: 0.552878518894243 .
Epoch: 031, Loss: 3.6133
Train Accuracy: 0.6622820919175911 . Test Accuracy: 0.6766421506467157 .
Epoch: 032, Loss: 2.7142
Train Accuracy: 0.7472266244057052 . Test Accuracy: 0.7438498605123003 .
Epoch: 033, Loss: 4.9442
Train Accuracy: 0.655229793977813 . Test Accuracy: 0.6700481866599036 .
Epoch: 034, Loss: 1.9909
Train Accuracy: 0.7289223454833598 . Test Accuracy: 0.7304083185391833 .
Epoch: 035, Loss: 3.0802
Train Accuracy: 0.6862916006339144 . Test Accuracy: 0.686279482627441 .
Epoch: 036, Loss: 2.9786
Train Accuracy: 0.7320919175911252 . Test Accuracy: 0.7248288105503424 .
Epoch: 037, Loss: 1.9850
Train Accuracy: 0.7007131537242473 . Test Accuracy: 0.7073294445853411 .
Epoch: 038, Loss: 3.4915
Train Accuracy: 0.681378763866878 . Test Accuracy: 0.6819680446360639 .
Epoch: 039, Loss: 2.4257
Train Accuracy: 0.7635499207606973 . Test Accuracy: 0.755262490489475 .
Epoch: 040, Loss: 2.7012
Train Accuracy: 0.6183042789223455 . Test Accuracy: 0.6233832107532336 .
Epoch: 041, Loss: 2.1345
Train Accuracy: 0.68486529318542 . Test Accuracy: 0.691858990616282 .
Epoch: 042, Loss: 3.0411
Train Accuracy: 0.7530903328050713 . Test Accuracy: 0.7501902104996195 .
Epoch: 043, Loss: 2.3290
Train Accuracy: 0.7460380348652932 . Test Accuracy: 0.7438498605123003 .
Epoch: 044, Loss: 1.6355
Train Accuracy: 0.7824881141045958 . Test Accuracy: 0.7846817144306366 .
Epoch: 045, Loss: 1.8552
Train Accuracy: 0.7472266244057052 . Test Accuracy: 0.7514582804970834 .
Epoch: 046, Loss: 2.9916
Train Accuracy: 0.74540412044374 . Test Accuracy: 0.7499365965001268 .
Epoch: 047, Loss: 1.9906
Train Accuracy: 0.7476228209191759 . Test Accuracy: 0.7446107025107785 .
Epoch: 048, Loss: 1.7028
Train Accuracy: 0.7282091917591125 . Test Accuracy: 0.7299010905401978 .
Epoch: 049, Loss: 2.6138
Train Accuracy: 0.726148969889065 . Test Accuracy: 0.7291402485417195 .
Epoch: 050, Loss: 2.0523
Train Accuracy: 0.7610935023771791 . Test Accuracy: 0.7651534364696931 .
Epoch: 051, Loss: 2.5033
Train Accuracy: 0.7316957210776546 . Test Accuracy: 0.7344661425310677 .
Epoch: 052, Loss: 2.1107
Train Accuracy: 0.7391442155309034 . Test Accuracy: 0.7397920365204159 .
Epoch: 053, Loss: 1.7962
Train Accuracy: 0.7208399366085578 . Test Accuracy: 0.7220390565559219 .
Epoch: 054, Loss: 1.5728
Train Accuracy: 0.7263074484944533 . Test Accuracy: 0.7197565305604869 .
Epoch: 055, Loss: 2.0867
Train Accuracy: 0.7503961965134707 . Test Accuracy: 0.7509510524980979 .
Epoch: 056, Loss: 1.7071
Train Accuracy: 0.7134706814580032 . Test Accuracy: 0.7220390565559219 .
Epoch: 057, Loss: 2.2497
Train Accuracy: 0.746513470681458 . Test Accuracy: 0.7481612985036774 .
Epoch: 058, Loss: 1.6381
Train Accuracy: 0.7450871632329635 . Test Accuracy: 0.7489221405021557 .
Epoch: 059, Loss: 2.1804
Train Accuracy: 0.7473851030110935 . Test Accuracy: 0.7537408064925184 .
Epoch: 060, Loss: 1.7475
Train Accuracy: 0.7184627575277338 . Test Accuracy: 0.7299010905401978 .
Epoch: 061, Loss: 2.0024
Train Accuracy: 0.7164817749603803 . Test Accuracy: 0.7162059345675881 .
Epoch: 062, Loss: 1.9603
Train Accuracy: 0.7439778129952457 . Test Accuracy: 0.7512046664975907 .
Epoch: 063, Loss: 1.9844
Train Accuracy: 0.7513470681458003 . Test Accuracy: 0.7547552624904895 .
Epoch: 064, Loss: 1.2897
Train Accuracy: 0.7478605388272583 . Test Accuracy: 0.7456251585087497 .
Epoch: 065, Loss: 2.3263
Train Accuracy: 0.7405705229793977 . Test Accuracy: 0.7390311945219377 .
Epoch: 066, Loss: 2.3476
Train Accuracy: 0.7083201267828844 . Test Accuracy: 0.7002282525995435 .
Epoch: 067, Loss: 1.4244
Train Accuracy: 0.7020602218700476 . Test Accuracy: 0.6981993406036013 .
Epoch: 068, Loss: 1.8484
Train Accuracy: 0.7530903328050713 . Test Accuracy: 0.7605883844788233 .
Epoch: 069, Loss: 1.9353
Train Accuracy: 0.7370047543581616 . Test Accuracy: 0.7423281765153437 .
Epoch: 070, Loss: 2.0585
Train Accuracy: 0.7485736925515055 . Test Accuracy: 0.7504438244991124 .
Epoch: 071, Loss: 1.5070
Train Accuracy: 0.7064183835182251 . Test Accuracy: 0.7004818665990362 .
Epoch: 072, Loss: 1.5660
Train Accuracy: 0.7707606973058637 . Test Accuracy: 0.7725082424549835 .
Epoch: 073, Loss: 1.3645
Train Accuracy: 0.7183042789223455 . Test Accuracy: 0.7235607405528786 .
Epoch: 074, Loss: 2.0699
Train Accuracy: 0.7587955625990491 . Test Accuracy: 0.7651534364696931 .
Epoch: 075, Loss: 1.7240
Train Accuracy: 0.7339936608557844 . Test Accuracy: 0.7339589145320822 .
Epoch: 076, Loss: 2.3821
Train Accuracy: 0.6682250396196513 . Test Accuracy: 0.6733451686533096 .
Epoch: 077, Loss: 1.6211
Train Accuracy: 0.7457210776545167 . Test Accuracy: 0.7491757545016485 .
Epoch: 078, Loss: 1.9058
Train Accuracy: 0.6961965134706815 . Test Accuracy: 0.7022571645954857 .
Epoch: 079, Loss: 1.9895
Train Accuracy: 0.7364500792393027 . Test Accuracy: 0.7372558965254882 .
Epoch: 080, Loss: 1.6059
Train Accuracy: 0.7005546751188589 . Test Accuracy: 0.7045396905909206 .
Epoch: 081, Loss: 1.4119
Train Accuracy: 0.7572900158478605 . Test Accuracy: 0.7595739284808521 .
Epoch: 082, Loss: 1.6939
Train Accuracy: 0.7795562599049128 . Test Accuracy: 0.7821455744357089 .
Epoch: 083, Loss: 1.5923
Train Accuracy: 0.7327258320126783 . Test Accuracy: 0.7314227745371544 .
Epoch: 084, Loss: 1.5504
Train Accuracy: 0.7536450079239303 . Test Accuracy: 0.7580522444838955 .
Epoch: 085, Loss: 1.6074
Train Accuracy: 0.6838351822503962 . Test Accuracy: 0.6913517626172965 .
Epoch: 086, Loss: 1.2307
Train Accuracy: 0.7481774960380349 . Test Accuracy: 0.7616028404767943 .
Epoch: 087, Loss: 1.3572
Train Accuracy: 0.7217115689381933 . Test Accuracy: 0.717474004565052 .
Epoch: 088, Loss: 1.4457
Train Accuracy: 0.7297147385103011 . Test Accuracy: 0.7281257925437484 .
Epoch: 089, Loss: 1.5707
Train Accuracy: 0.7496038034865293 . Test Accuracy: 0.7504438244991124 .
Epoch: 090, Loss: 1.6337
Train Accuracy: 0.7539619651347068 . Test Accuracy: 0.764138980471722 .
Epoch: 091, Loss: 1.5653
Train Accuracy: 0.7144215530903328 . Test Accuracy: 0.7189956885620086 .
Epoch: 092, Loss: 1.3315
Train Accuracy: 0.6091917591125198 . Test Accuracy: 0.6198326147603348 .
Epoch: 093, Loss: 1.3703
Train Accuracy: 0.6792393026941362 . Test Accuracy: 0.6728379406543241 .
Epoch: 094, Loss: 1.6818
Train Accuracy: 0.7439778129952457 . Test Accuracy: 0.7501902104996195 .
Epoch: 095, Loss: 1.6053
Train Accuracy: 0.6333597464342314 . Test Accuracy: 0.6408825767182349 .
Epoch: 096, Loss: 1.5899
Train Accuracy: 0.7404912836767037 . Test Accuracy: 0.7359878265280243 .
Epoch: 097, Loss: 2.3347
Train Accuracy: 0.7718700475435816 . Test Accuracy: 0.7788485924423029 .
Epoch: 098, Loss: 1.2340
Train Accuracy: 0.7243264659270998 . Test Accuracy: 0.7167131625665737 .
Epoch: 099, Loss: 1.5172
Train Accuracy: 0.5242472266244057 . Test Accuracy: 0.5320821709358357 .
Epoch: 100, Loss: 1.2344
Train Accuracy: 0.7425515055467512 . Test Accuracy: 0.7377631245244738 .
Epoch: 101, Loss: 1.4465
Train Accuracy: 0.690095087163233 . Test Accuracy: 0.6910981486178037 .
Epoch: 102, Loss: 1.7077
Train Accuracy: 0.7241679873217116 . Test Accuracy: 0.7276185645447629 .
Epoch: 103, Loss: 1.3107
Train Accuracy: 0.7002377179080824 . Test Accuracy: 0.7027643925944712 .
Epoch: 104, Loss: 1.6042
Train Accuracy: 0.7023771790808241 . Test Accuracy: 0.6969312706061375 .
Epoch: 105, Loss: 1.7569
Train Accuracy: 0.7549920760697306 . Test Accuracy: 0.7453715445092569 .
Epoch: 106, Loss: 1.9007
Train Accuracy: 0.6857369255150555 . Test Accuracy: 0.6804463606391072 .
Epoch: 107, Loss: 2.4890
Train Accuracy: 0.7478605388272583 . Test Accuracy: 0.749682982500634 .
Epoch: 108, Loss: 1.9098
Train Accuracy: 0.7011093502377179 . Test Accuracy: 0.6951559726096881 .
Epoch: 109, Loss: 1.3094
Train Accuracy: 0.7167194928684627 . Test Accuracy: 0.7139234085721532 .
Epoch: 110, Loss: 1.5052
Train Accuracy: 0.7524564183835182 . Test Accuracy: 0.7572914024854172 .
Epoch: 111, Loss: 1.4712
Train Accuracy: 0.5678288431061806 . Test Accuracy: 0.5731676388536647 .
Epoch: 112, Loss: 1.7028
Train Accuracy: 0.7194928684627575 . Test Accuracy: 0.7245751965508496 .
Epoch: 113, Loss: 1.7264
Train Accuracy: 0.6928684627575278 . Test Accuracy: 0.7027643925944712 .
Epoch: 114, Loss: 1.2611
Train Accuracy: 0.7300316957210776 . Test Accuracy: 0.7189956885620086 .
Epoch: 115, Loss: 2.7755
Train Accuracy: 0.5583201267828843 . Test Accuracy: 0.5645447628709105 .
Epoch: 116, Loss: 1.5589
Train Accuracy: 0.709904912836767 . Test Accuracy: 0.7167131625665737 .
Epoch: 117, Loss: 1.3587
Train Accuracy: 0.7306656101426308 . Test Accuracy: 0.7321836165356328 .
Epoch: 118, Loss: 1.0785
Train Accuracy: 0.7574484944532488 . Test Accuracy: 0.7517118944965762 .
Epoch: 119, Loss: 1.5245
Train Accuracy: 0.7041996830427892 . Test Accuracy: 0.7124017245751966 .
Epoch: 120, Loss: 1.3863
Train Accuracy: 0.7216323296354992 . Test Accuracy: 0.7288866345422267 .
Epoch: 121, Loss: 1.9658
Train Accuracy: 0.6020602218700476 . Test Accuracy: 0.6028404767943191 .
Epoch: 122, Loss: 2.3056
Train Accuracy: 0.7370839936608558 . Test Accuracy: 0.7370022825259954 .
Epoch: 123, Loss: 1.8290
Train Accuracy: 0.7187004754358162 . Test Accuracy: 0.7091047425817906 .
Epoch: 124, Loss: 1.2217
Train Accuracy: 0.7616481774960381 . Test Accuracy: 0.7605883844788233 .
Epoch: 125, Loss: 1.6795
Train Accuracy: 0.6659270998415214 . Test Accuracy: 0.6746132386507735 .
Epoch: 126, Loss: 1.3714
Train Accuracy: 0.6885103011093502 . Test Accuracy: 0.6905909206188181 .
Epoch: 127, Loss: 1.7124
Train Accuracy: 0.7361331220285261 . Test Accuracy: 0.7337053005325894 .
Epoch: 128, Loss: 1.6824
Train Accuracy: 0.775039619651347 . Test Accuracy: 0.7745371544509256 .
Epoch: 129, Loss: 2.0378
Train Accuracy: 0.7564976228209191 . Test Accuracy: 0.747400456505199 .
Epoch: 130, Loss: 2.5637
Train Accuracy: 0.6992076069730586 . Test Accuracy: 0.705300532589399 .
Epoch: 131, Loss: 3.3471
Train Accuracy: 0.7721870047543582 . Test Accuracy: 0.7763124524473751 .
Epoch: 132, Loss: 1.6103
Train Accuracy: 0.7565768621236133 . Test Accuracy: 0.7539944204920112 .
Epoch: 133, Loss: 2.5757
Train Accuracy: 0.7410459587955626 . Test Accuracy: 0.7402992645194014 .
Epoch: 134, Loss: 1.4756
Train Accuracy: 0.6760697305863709 . Test Accuracy: 0.6756276946487446 .
Epoch: 135, Loss: 2.2291
Train Accuracy: 0.7374009508716324 . Test Accuracy: 0.7324372305351255 .
Epoch: 136, Loss: 2.1387
Train Accuracy: 0.645958795562599 . Test Accuracy: 0.6421506467156987 .
Epoch: 137, Loss: 1.7185
Train Accuracy: 0.6905705229793978 . Test Accuracy: 0.6814608166370784 .
Epoch: 138, Loss: 1.3057
Train Accuracy: 0.7797147385103012 . Test Accuracy: 0.7788485924423029 .
Epoch: 139, Loss: 1.5246
Train Accuracy: 0.6276545166402535 . Test Accuracy: 0.6330205427339589 .
Epoch: 140, Loss: 1.1474
Train Accuracy: 0.5233755942947702 . Test Accuracy: 0.5237129089525742 .
Epoch: 141, Loss: 5.2866
Train Accuracy: 0.755229793977813 . Test Accuracy: 0.7491757545016485 .
Epoch: 142, Loss: 2.8881
Train Accuracy: 0.7481774960380349 . Test Accuracy: 0.7547552624904895 .
Epoch: 143, Loss: 1.7474
Train Accuracy: 0.754041204437401 . Test Accuracy: 0.7633781384732438 .
Epoch: 144, Loss: 2.0925
Train Accuracy: 0.7656101426307449 . Test Accuracy: 0.7742835404514329 .
Epoch: 145, Loss: 4.6637
Train Accuracy: 0.7358954041204437 . Test Accuracy: 0.7357342125285316 .
Epoch: 146, Loss: 1.9865
Train Accuracy: 0.7420760697305864 . Test Accuracy: 0.7453715445092569 .
Epoch: 147, Loss: 1.0630
Train Accuracy: 0.786053882725832 . Test Accuracy: 0.7803702764392595 .
Epoch: 148, Loss: 1.4938
Train Accuracy: 0.7146592709984152 . Test Accuracy: 0.7129089525741821 .
Epoch: 149, Loss: 3.5124
Train Accuracy: 0.6541996830427892 . Test Accuracy: 0.6614253106771494 .
Epoch: 150, Loss: 2.0871
Train Accuracy: 0.7094294770206022 . Test Accuracy: 0.6969312706061375 .
Epoch: 151, Loss: 2.3136
Train Accuracy: 0.7194136291600634 . Test Accuracy: 0.7131625665736748 .
Epoch: 152, Loss: 2.0930
Train Accuracy: 0.5921553090332805 . Test Accuracy: 0.5822977428354045 .
Epoch: 153, Loss: 1.9939
Train Accuracy: 0.7217908082408875 . Test Accuracy: 0.7159523205680953 .
Epoch: 154, Loss: 4.2513
Train Accuracy: 0.5847860538827259 . Test Accuracy: 0.5736748668526502 .
Epoch: 155, Loss: 2.4594
Train Accuracy: 0.7649762282091918 . Test Accuracy: 0.7605883844788233 .
Epoch: 156, Loss: 2.7144
Train Accuracy: 0.7905705229793978 . Test Accuracy: 0.7933045904133909 .
Epoch: 157, Loss: 2.8539
Train Accuracy: 0.7767036450079239 . Test Accuracy: 0.7818919604362161 .
Epoch: 158, Loss: 2.0907
Train Accuracy: 0.7580031695721078 . Test Accuracy: 0.7560233324879533 .
Epoch: 159, Loss: 1.4725
Train Accuracy: 0.7675911251980982 . Test Accuracy: 0.7679431904641136 .
Epoch: 160, Loss: 1.8810
Train Accuracy: 0.7767036450079239 . Test Accuracy: 0.7839208724321582 .
Epoch: 161, Loss: 2.0854
Train Accuracy: 0.7321711568938193 . Test Accuracy: 0.7349733705300533 .
Epoch: 162, Loss: 3.4171
Train Accuracy: 0.721473851030111 . Test Accuracy: 0.7195029165609942 .
Epoch: 163, Loss: 1.7088
Train Accuracy: 0.7343106180665611 . Test Accuracy: 0.7339589145320822 .
Epoch: 164, Loss: 1.7724
Train Accuracy: 0.4794770206022187 . Test Accuracy: 0.4889677910220644 .
Epoch: 165, Loss: 1.1750
Train Accuracy: 0.7583201267828843 . Test Accuracy: 0.7583058584833883 .
Epoch: 166, Loss: 1.4929
Train Accuracy: 0.6905705229793978 . Test Accuracy: 0.7050469185899062 .
Epoch: 167, Loss: 2.0151
Train Accuracy: 0.7437400950871632 . Test Accuracy: 0.7481612985036774 .
Epoch: 168, Loss: 4.1040
Train Accuracy: 0.7595879556259905 . Test Accuracy: 0.7699721024600558 .
Epoch: 169, Loss: 1.8940
Train Accuracy: 0.5621236133122028 . Test Accuracy: 0.5622622368754755 .
Epoch: 170, Loss: 4.2639
Train Accuracy: 0.7167987321711569 . Test Accuracy: 0.7172203905655592 .
Epoch: 171, Loss: 3.1969
Train Accuracy: 0.7692551505546751 . Test Accuracy: 0.7694648744610703 .
Epoch: 172, Loss: 1.3067
Train Accuracy: 0.7636291600633914 . Test Accuracy: 0.7646462084707076 .
Epoch: 173, Loss: 2.4706
Train Accuracy: 0.7719492868462757 . Test Accuracy: 0.7651534364696931 .
Epoch: 174, Loss: 3.3058
Train Accuracy: 0.7392234548335974 . Test Accuracy: 0.7354805985290388 .
Epoch: 175, Loss: 3.3684
Train Accuracy: 0.6032488114104596 . Test Accuracy: 0.6046157747907684 .
Epoch: 176, Loss: 3.5566
Train Accuracy: 0.7362916006339144 . Test Accuracy: 0.7347197565305605 .
Epoch: 177, Loss: 2.2526
Train Accuracy: 0.6891442155309033 . Test Accuracy: 0.6834897286330205 .
Epoch: 178, Loss: 2.0195
Train Accuracy: 0.7512678288431062 . Test Accuracy: 0.7423281765153437 .
Epoch: 179, Loss: 2.7985
Train Accuracy: 0.7477812995245642 . Test Accuracy: 0.7392848085214304 .
Epoch: 180, Loss: 2.0457
Train Accuracy: 0.7549920760697306 . Test Accuracy: 0.7547552624904895 .
Epoch: 181, Loss: 2.7411
Train Accuracy: 0.7496038034865293 . Test Accuracy: 0.7453715445092569 .
Epoch: 182, Loss: 1.8084
Train Accuracy: 0.6645800316957211 . Test Accuracy: 0.668526502662947 .
Epoch: 183, Loss: 1.5359
Train Accuracy: 0.7946117274167988 . Test Accuracy: 0.7981232564037535 .
Epoch: 184, Loss: 2.7700
Train Accuracy: 0.7392234548335974 . Test Accuracy: 0.7443570885112858 .
Epoch: 185, Loss: 1.8069
Train Accuracy: 0.7648177496038034 . Test Accuracy: 0.7727618564544763 .
Epoch: 186, Loss: 1.5763
Train Accuracy: 0.7865293185419968 . Test Accuracy: 0.7884859244230281 .
Epoch: 187, Loss: 1.6727
Train Accuracy: 0.7441362916006339 . Test Accuracy: 0.7504438244991124 .
Epoch: 188, Loss: 1.7070
Train Accuracy: 0.7320126782884311 . Test Accuracy: 0.7339589145320822 .
Epoch: 189, Loss: 1.4043
Train Accuracy: 0.7766244057052298 . Test Accuracy: 0.7798630484402739 .
Epoch: 190, Loss: 2.4622
Train Accuracy: 0.739540412044374 . Test Accuracy: 0.7446107025107785 .
Epoch: 191, Loss: 1.4305
Train Accuracy: 0.708161648177496 . Test Accuracy: 0.7098655845802688 .
Epoch: 192, Loss: 2.3568
Train Accuracy: 0.7514263074484945 . Test Accuracy: 0.7542480344915039 .
Epoch: 193, Loss: 1.3124
Train Accuracy: 0.7316957210776546 . Test Accuracy: 0.7359878265280243 .
Epoch: 194, Loss: 2.2812
Train Accuracy: 0.6799524564183835 . Test Accuracy: 0.6723307126553386 .
Epoch: 195, Loss: 1.5512
Train Accuracy: 0.7306656101426308 . Test Accuracy: 0.7372558965254882 .
Epoch: 196, Loss: 1.7271
Train Accuracy: 0.775594294770206 . Test Accuracy: 0.7712401724575196 .
Epoch: 197, Loss: 1.5465
Train Accuracy: 0.7643423137876386 . Test Accuracy: 0.7636317524727365 .
Epoch: 198, Loss: 2.0448
Train Accuracy: 0.7999207606973059 . Test Accuracy: 0.7993913264012174 .
Epoch: 199, Loss: 1.7823
Train Accuracy: 0.7759904912836767 . Test Accuracy: 0.7727618564544763 .
Epoch: 200, Loss: 1.6234
Train Accuracy: 0.7733755942947702 . Test Accuracy: 0.7722546284554908 .

Best Case:
Epoch: 198, Loss: 2.0448
Train Accuracy: 0.7999207606973059 . Test Accuracy: 0.7993913264012174 .
'''
