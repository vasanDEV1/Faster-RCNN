import re
import matplotlib.pyplot as plt


log_text = """
Epoch [0] Loss: 0.0965
Model saved: fasterrcnn_resnet50_epoch_1.pth
Epoch [1] Loss: 0.0991
Epoch [2] Loss: 0.0741
Epoch [3] Loss: 0.0441
Epoch [4] Loss: 0.0728
Epoch [5] Loss: 0.0605
Epoch [6] Loss: 0.0674
Epoch [7] Loss: 0.0681
Epoch [8] Loss: 0.0492
Epoch [9] Loss: 0.0552
Epoch [10] Loss: 0.0403
Model saved: fasterrcnn_resnet50_epoch_11.pth
Epoch [11] Loss: 0.0507
Epoch [12] Loss: 0.0394
Epoch [13] Loss: 0.0427
Epoch [14] Loss: 0.0415
Epoch [15] Loss: 0.0501
Epoch [16] Loss: 0.0512
Epoch [17] Loss: 0.0626
Epoch [18] Loss: 0.0458
Epoch [19] Loss: 0.0446
Epoch [20] Loss: 0.0623
Model saved: fasterrcnn_resnet50_epoch_21.pth
Epoch [21] Loss: 0.0555
Epoch [22] Loss: 0.0603
Epoch [23] Loss: 0.0471
Epoch [24] Loss: 0.0625
Epoch [25] Loss: 0.0448
Epoch [26] Loss: 0.0660
Epoch [27] Loss: 0.0461
Epoch [28] Loss: 0.0534
Epoch [29] Loss: 0.0511
Epoch [30] Loss: 0.0517
Model saved: fasterrcnn_resnet50_epoch_31.pth
Epoch [31] Loss: 0.0379
Epoch [32] Loss: 0.0436
Epoch [33] Loss: 0.0525
Epoch [34] Loss: 0.0425
Epoch [35] Loss: 0.0461
Epoch [36] Loss: 0.0420
Epoch [37] Loss: 0.0515
Epoch [38] Loss: 0.0505
Epoch [39] Loss: 0.0869
Epoch [40] Loss: 0.0459
Model saved: fasterrcnn_resnet50_epoch_41.pth
Epoch [41] Loss: 0.0373
Epoch [42] Loss: 0.0467
Epoch [43] Loss: 0.0406
Epoch [44] Loss: 0.0515
Epoch [45] Loss: 0.0742
Epoch [46] Loss: 0.0737
Epoch [47] Loss: 0.0690
Epoch [48] Loss: 0.0427
Epoch [49] Loss: 0.0407
Epoch [50] Loss: 0.0580
Model saved: fasterrcnn_resnet50_epoch_51.pth
Epoch [51] Loss: 0.0359
Epoch [52] Loss: 0.0505
Epoch [53] Loss: 0.0416
Epoch [54] Loss: 0.0526
Epoch [55] Loss: 0.0515
Epoch [56] Loss: 0.0565
Epoch [57] Loss: 0.0374
Epoch [58] Loss: 0.0434
Epoch [59] Loss: 0.0453
Epoch [60] Loss: 0.0433
Model saved: fasterrcnn_resnet50_epoch_61.pth
Epoch [61] Loss: 0.0815
Epoch [62] Loss: 0.0665
Epoch [63] Loss: 0.0615
Epoch [64] Loss: 0.0456
Epoch [65] Loss: 0.0563
Epoch [66] Loss: 0.0585
Epoch [67] Loss: 0.0586
Epoch [68] Loss: 0.0568
Epoch [69] Loss: 0.0447
Epoch [70] Loss: 0.0601
Model saved: fasterrcnn_resnet50_epoch_71.pth
Epoch [71] Loss: 0.0462
Epoch [72] Loss: 0.0379
Epoch [73] Loss: 0.0558
Epoch [74] Loss: 0.0476
Epoch [75] Loss: 0.0543
Epoch [76] Loss: 0.0456
Epoch [77] Loss: 0.0486
Epoch [78] Loss: 0.0651
Epoch [79] Loss: 0.0453
Epoch [80] Loss: 0.0648
Model saved: fasterrcnn_resnet50_epoch_81.pth
Epoch [81] Loss: 0.0503
Epoch [82] Loss: 0.0470
Epoch [83] Loss: 0.0516
Epoch [84] Loss: 0.0436
Epoch [85] Loss: 0.0583
Epoch [86] Loss: 0.0537
Epoch [87] Loss: 0.0517
Epoch [88] Loss: 0.0441
Epoch [89] Loss: 0.0386
Epoch [90] Loss: 0.0423
Model saved: fasterrcnn_resnet50_epoch_91.pth
Epoch [91] Loss: 0.0489
Epoch [92] Loss: 0.0568
Epoch [93] Loss: 0.0398
Epoch [94] Loss: 0.0384
Epoch [95] Loss: 0.0384
Epoch [96] Loss: 0.0517
Epoch [97] Loss: 0.0612
Epoch [98] Loss: 0.0631
Epoch [99] Loss: 0.0670
Epoch [100] Loss: 0.0529
Model saved: fasterrcnn_resnet50_epoch_101.pth
Epoch [101] Loss: 0.0543
Epoch [102] Loss: 0.0440
Epoch [103] Loss: 0.0361
Epoch [104] Loss: 0.0406
Epoch [105] Loss: 0.0433
Epoch [106] Loss: 0.0498
Epoch [107] Loss: 0.0535
Epoch [108] Loss: 0.0547
Epoch [109] Loss: 0.0324
Epoch [110] Loss: 0.0730
Model saved: fasterrcnn_resnet50_epoch_111.pth
Epoch [111] Loss: 0.0613
Epoch [112] Loss: 0.0561
Epoch [113] Loss: 0.0370
Epoch [114] Loss: 0.0393
Epoch [115] Loss: 0.0485
Epoch [116] Loss: 0.0396
Epoch [117] Loss: 0.0451
Epoch [118] Loss: 0.0529
Epoch [119] Loss: 0.0436
Epoch [120] Loss: 0.0447
Model saved: fasterrcnn_resnet50_epoch_121.pth
Epoch [121] Loss: 0.0554
Epoch [122] Loss: 0.0467
Epoch [123] Loss: 0.0450
Epoch [124] Loss: 0.0592
Epoch [125] Loss: 0.0540
Epoch [126] Loss: 0.0483
Epoch [127] Loss: 0.0631
Epoch [128] Loss: 0.0488
Epoch [129] Loss: 0.0291
Epoch [130] Loss: 0.0362
Model saved: fasterrcnn_resnet50_epoch_131.pth
Epoch [131] Loss: 0.0511
Epoch [132] Loss: 0.0627
Epoch [133] Loss: 0.0511
Epoch [134] Loss: 0.0407
Epoch [135] Loss: 0.0530
Epoch [136] Loss: 0.0558
Epoch [137] Loss: 0.0367
Epoch [138] Loss: 0.0516
Epoch [139] Loss: 0.0441
Epoch [140] Loss: 0.0330
Model saved: fasterrcnn_resnet50_epoch_141.pth
Epoch [141] Loss: 0.0428
Epoch [142] Loss: 0.0535
Epoch [143] Loss: 0.0405
Epoch [144] Loss: 0.0705
Epoch [145] Loss: 0.0387
Epoch [146] Loss: 0.0461
Epoch [147] Loss: 0.0680
Epoch [148] Loss: 0.0512
Epoch [149] Loss: 0.0412
Epoch [150] Loss: 0.0648
Model saved: fasterrcnn_resnet50_epoch_151.pth
Epoch [151] Loss: 0.0490
Epoch [152] Loss: 0.0403
Epoch [153] Loss: 0.0488
Epoch [154] Loss: 0.0438
Epoch [155] Loss: 0.0345
Epoch [156] Loss: 0.0365
Epoch [157] Loss: 0.0568
Epoch [158] Loss: 0.0405
Epoch [159] Loss: 0.0463
Epoch [160] Loss: 0.0492
Model saved: fasterrcnn_resnet50_epoch_161.pth
Epoch [161] Loss: 0.0472
Epoch [162] Loss: 0.0396
Epoch [163] Loss: 0.0452
Epoch [164] Loss: 0.0617
Epoch [165] Loss: 0.0401
Epoch [166] Loss: 0.0660
Epoch [167] Loss: 0.0645
Epoch [168] Loss: 0.0641
Epoch [169] Loss: 0.0426
Epoch [170] Loss: 0.0482
Model saved: fasterrcnn_resnet50_epoch_171.pth
Epoch [171] Loss: 0.0611
Epoch [172] Loss: 0.0622
Epoch [173] Loss: 0.0471
Epoch [174] Loss: 0.0400
Epoch [175] Loss: 0.0564
Epoch [176] Loss: 0.0495
Epoch [177] Loss: 0.0609
Epoch [178] Loss: 0.0629
Epoch [179] Loss: 0.0471
Epoch [180] Loss: 0.0384
Model saved: fasterrcnn_resnet50_epoch_181.pth
Epoch [181] Loss: 0.0600
Epoch [182] Loss: 0.0429
Epoch [183] Loss: 0.0560
Epoch [184] Loss: 0.0398
Epoch [185] Loss: 0.0490
Epoch [186] Loss: 0.0612
Epoch [187] Loss: 0.0422
Epoch [188] Loss: 0.0826
Epoch [189] Loss: 0.0479
Epoch [190] Loss: 0.0432
Model saved: fasterrcnn_resnet50_epoch_191.pth
Epoch [191] Loss: 0.0732
Epoch [192] Loss: 0.0615
Epoch [193] Loss: 0.0530
Epoch [194] Loss: 0.0415
Epoch [195] Loss: 0.0318
Epoch [196] Loss: 0.0437
Epoch [197] Loss: 0.0479
Epoch [198] Loss: 0.0304
Epoch [199] Loss: 0.0454
Epoch [200] Loss: 0.0477
Model saved: fasterrcnn_resnet50_epoch_201.pth
Epoch [201] Loss: 0.0516
Epoch [202] Loss: 0.0637
Epoch [203] Loss: 0.0423
Epoch [204] Loss: 0.0555
Epoch [205] Loss: 0.0544
Epoch [206] Loss: 0.0441
Epoch [207] Loss: 0.0508
Epoch [208] Loss: 0.0361
Epoch [209] Loss: 0.0549
Epoch [210] Loss: 0.0479
Model saved: fasterrcnn_resnet50_epoch_211.pth
Epoch [211] Loss: 0.0353
Epoch [212] Loss: 0.0336
Epoch [213] Loss: 0.0469
Epoch [214] Loss: 0.0364
Epoch [215] Loss: 0.0760
Epoch [216] Loss: 0.0473
Epoch [217] Loss: 0.0570
Epoch [218] Loss: 0.0634
Epoch [219] Loss: 0.0398
Epoch [220] Loss: 0.0701
Model saved: fasterrcnn_resnet50_epoch_221.pth
"""


pattern = r"Epoch \[(\d+)\] Loss: ([0-9.]+)"


matches = re.findall(pattern, log_text)


epochs = [int(epoch) for epoch, loss in matches]
losses = [float(loss) for epoch, loss in matches]


plt.figure(figsize=(12, 6))
plt.plot(epochs, losses, marker='o', linestyle='-', color='blue')
plt.title("Epoch vs Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss Rate")
plt.grid(True)
plt.tight_layout()
plt.show()
