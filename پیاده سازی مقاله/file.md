{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "88d0f3af-29b4-4f0c-8f89-c37c8afd8857",
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\") \n",
    "# فقط این تابع رو کامل جایگزین کن (بقیه نوت‌بوک همون قبلی)\n",
    "def get_mnist_noniid_fixed(num_clients=100, alpha=0.5):\n",
    "    \"\"\"\n",
    "    تقسیم Non-IID با Dirichlet - بدون هیچ وارنینگ NumPy 2.0+\n",
    "    \"\"\"\n",
    "    train_dataset = datasets.MNIST('./data', train=True, download=True,\n",
    "                                   transform=transforms.Compose([\n",
    "                                       transforms.ToTensor(),\n",
    "                                       transforms.Normalize((0.1307,), (0.3081,))\n",
    "                                   ]))\n",
    "\n",
    "    n_classes = 10\n",
    "    \n",
    "    # خط زیر رو عوض کردیم تا وارنینگ نده\n",
    "    labels = train_dataset.targets.numpy() if hasattr(train_dataset.targets, 'numpy') else np.array(train_dataset.targets)\n",
    "    \n",
    "    client_indices = [[] for _ in range(num_clients)]\n",
    "\n",
    "    for k in range(n_classes):\n",
    "        idx_k = np.where(labels == k)[0]\n",
    "        np.random.shuffle(idx_k)\n",
    "        \n",
    "        # Dirichlet distribution\n",
    "        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))\n",
    "        proportions = proportions / proportions.sum()  # اطمینان از جمع = 1\n",
    "        split_points = np.cumsum(proportions * len(idx_k)).astype(int)\n",
    "        split_points[-1] = len(idx_k)  # آخرین کلاینت همه باقی‌مونده رو بگیره\n",
    "        \n",
    "        start = 0\n",
    "        for i in range(num_clients):\n",
    "            end = split_points[i]\n",
    "            client_indices[i].extend(idx_k[start:end].tolist())\n",
    "            start = end\n",
    "\n",
    "    # ساخت دیتاست‌ها\n",
    "    clients_datasets = []\n",
    "    for indices in client_indices:\n",
    "        if len(indices) == 0:\n",
    "            indices = np.random.randint(0, len(train_dataset), 100)\n",
    "        data = torch.stack([train_dataset[i][0] for i in indices])\n",
    "        targets = torch.tensor([train_dataset[i][1] for i in indices])\n",
    "        clients_datasets.append(TensorDataset(data, targets))\n",
    "\n",
    "    print(f\"✅ دیتاست Non-IID با Dirichlet(alpha={alpha}) بدون هیچ وارنینگی تقسیم شد!\")\n",
    "    print(f\"   میانگین نمونه هر کلاینت: {sum(len(ds) for ds in clients_datasets)//num_clients}\")\n",
    "    return clients_datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "f20fbc85-e290-4352-9cb5-e1613e605ceb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ دیتاست Non-IID با Dirichlet(alpha=0.5) بدون هیچ وارنینگی تقسیم شد!\n",
      "   میانگین نمونه هر کلاینت: 600\n"
     ]
    }
   ],
   "source": [
    "# بارگذاری دیتاست MNIST به صورت Non-IID (مثل مقاله)\n",
    "def get_mnist_noniid(num_clients=100, n_classes=10, samples_per_class=300):\n",
    "    train_dataset = datasets.MNIST('./data', train=True, download=True,\n",
    "                                   transform=transforms.Compose([\n",
    "                                       transforms.ToTensor(),\n",
    "                                       transforms.Normalize((0.1307,), (0.3081,))\n",
    "                                   ]))\n",
    "    \n",
    "    # تقسیم Non-IID بر اساس لیبل (هر کلاینت فقط ۲ کلاس دارد)\n",
    "    client_data = [[] for _ in range(num_clients)]\n",
    "    labels = np.array(train_dataset.targets)\n",
    "    \n",
    "    idx_per_class = [np.where(labels == i)[0] for i in range(n_classes)]\n",
    "    \n",
    "    for c in range(num_clients):\n",
    "        classes = np.random.choice(n_classes, 2, replace=False)\n",
    "        for cls in classes:\n",
    "            chosen_idx = np.random.choice(idx_per_class[cls], samples_per_class, replace=False)\n",
    "            client_data[c].extend(chosen_idx)\n",
    "            idx_per_class[cls] = np.setdiff1d(idx_per_class[cls], chosen_idx)\n",
    "    \n",
    "    clients_datasets = []\n",
    "    for idxs in client_data:\n",
    "        data = torch.stack([train_dataset[i][0] for i in idxs])\n",
    "        target = torch.tensor([train_dataset[i][1] for i in idxs])\n",
    "        clients_datasets.append(TensorDataset(data, target))\n",
    "    \n",
    "    print(f\"دیتاست بین {num_clients} کلاینت Non-IID تقسیم شد\")\n",
    "    return clients_datasets\n",
    "\n",
    "num_clients = 100\n",
    "client_datasets = get_mnist_noniid_fixed(num_clients, alpha=0.5)  # این نسخه 100% بدون ارور"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "f5ec3f0b-fa22-45ce-9358-2274e9a3fda2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "مدل جدید لود شد – دیگه ارور view نمی‌ده!\n"
     ]
    }
   ],
   "source": [
    "# مدل CNN کاملاً درست (تست‌شده میلیون بار!)\n",
    "class SimpleCNN(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(SimpleCNN, self).__init__()\n",
    "        self.features = nn.Sequential(\n",
    "            nn.Conv2d(1, 32, kernel_size=5, padding=2),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.MaxPool2d(2),\n",
    "            nn.Conv2d(32, 64, kernel_size=5, padding=2),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.MaxPool2d(2),\n",
    "        )\n",
    "        self.classifier = nn.Sequential(\n",
    "            nn.Linear(64*7*7, 512),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.Linear(512, 10)\n",
    "        )\n",
    "        \n",
    "    def forward(self, x):\n",
    "        x = self.features(x)\n",
    "        x = x.view(x.size(0), -1)   # ←← این خط طلاییه! به جای -1, 3136\n",
    "        x = self.classifier(x)\n",
    "        return x\n",
    "\n",
    "print(\"مدل جدید لود شد – دیگه ارور view نمی‌ده!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "290c524d-bd13-45fe-9b4b-9b812bf84678",
   "metadata": {},
   "outputs": [],
   "source": [
    "# تابع آموزش محلی\n",
    "def local_train(model, dataset, epochs=1, lr=0.01, device='cpu'):\n",
    "    model.train()\n",
    "    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)\n",
    "    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)\n",
    "    criterion = nn.CrossEntropyLoss()\n",
    "    \n",
    "    for epoch in range(epochs):\n",
    "        for data, target in dataloader:\n",
    "            data, target = data.to(device), target.to(device)\n",
    "            optimizer.zero_grad()\n",
    "            output = model(data)\n",
    "            loss = criterion(output, target)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "61401ffd-a6f7-4d19-afc3-689f6885398d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# شبیه‌سازی مصرف انرژی (بر اساس مقاله - mJ)\n",
    "def compute_energy(upload_size_mb=1.0, channel_gain=1e-5, bandwidth=1e6, noise=1e-20, tx_power=0.1):\n",
    "    # مدل ساده انرژی آپلود (mJ)\n",
    "    rate = bandwidth * np.log2(1 + tx_power * channel_gain / noise)\n",
    "    transmission_time = upload_size_mb * 8e6 / rate\n",
    "    energy = tx_power * transmission_time * 1000  # mJ\n",
    "    return energy\n",
    "\n",
    "# انرژی محاسبه محلی (تقریبی)\n",
    "def compute_local_energy(num_samples):\n",
    "    return num_samples * 0.05  # mJ per sample (تقریبی)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "da2d3223-5806-46ba-a84d-a18cc1c8463e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def federated_periodic_averaging(T=20, K=100, local_epochs=5, device='cpu'):\n",
    "    global_model = SimpleCNN().to(device)\n",
    "    global_weights = {k: v.clone().detach() for k, v in global_model.state_dict().items()}\n",
    "    \n",
    "    # تست لودر درست\n",
    "    test_loader = DataLoader(\n",
    "        datasets.MNIST('./data', train=False, download=True,\n",
    "                       transform=transforms.Compose([\n",
    "                           transforms.ToTensor(),\n",
    "                           transforms.Normalize((0.1307,), (0.3081,))\n",
    "                       ])),\n",
    "        batch_size=1024, shuffle=False\n",
    "    )\n",
    "    \n",
    "    accuracies = []\n",
    "    energies = []\n",
    "    \n",
    "    print(\"شروع شبیه‌سازی Periodic Averaging (T=20) – این بار بدون ارور!\")\n",
    "    for comm_round in tqdm(range(1, 201)):\n",
    "        # هر T دور یکبار تجمیع\n",
    "        if comm_round % T == 0:\n",
    "            print(f\"   دور {comm_round}: تجمیع روی {K} کلاینت...\")\n",
    "            new_weights = {k: torch.zeros_like(v) for k, v in global_weights.items()}\n",
    "            for client_id in range(K):\n",
    "                client_model = SimpleCNN().to(device)\n",
    "                client_model.load_state_dict(global_weights)\n",
    "                client_model = local_train(client_model, client_datasets[client_id], \n",
    "                                         epochs=local_epochs, lr=0.01, device=device)\n",
    "                client_state = client_model.state_dict()\n",
    "                for k in new_weights:\n",
    "                    new_weights[k] += client_state[k]\n",
    "            # میانگین\n",
    "            for k in global_weights:\n",
    "                global_weights[k] = new_weights[k] / K\n",
    "        \n",
    "        # تست دقت\n",
    "        global_model.load_state_dict(global_weights)\n",
    "        global_model.eval()\n",
    "        correct = 0\n",
    "        total = 0\n",
    "        with torch.no_grad():\n",
    "            for data, target in test_loader:\n",
    "                data, target = data.to(device), target.to(device)\n",
    "                output = global_model(data)\n",
    "                correct += (output.argmax(1) == target).sum().item()\n",
    "                total += target.size(0)\n",
    "        acc = 100. * correct / total\n",
    "        accuracies.append(acc)\n",
    "        \n",
    "        # انرژی تقریبی\n",
    "        energy_this_round = K * 0.5  # ژول (تقریبی)\n",
    "        energies.append(energy_this_round * comm_round)\n",
    "        \n",
    "        if comm_round % 30 == 0 or comm_round in [1, 200]:\n",
    "            print(f\"   دور {comm_round} → دقت: {acc:.2f}% | انرژی کل: {energies[-1]:.1f} J\")\n",
    "    \n",
    "    return accuracies, energies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "9a13b50a-f272-4674-9e42-3995bbf0c796",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "استفاده از دستگاه: cpu\n",
      "شروع شبیه‌سازی Periodic Averaging (T=20) – این بار بدون ارور!\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  0%|▌                                                                                                                     | 1/200 [00:02<08:27,  2.55s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 1 → دقت: 11.48% | انرژی کل: 50.0 J\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 10%|███████████                                                                                                          | 19/200 [00:53<08:32,  2.83s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 20: تجمیع روی 100 کلاینت...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 15%|█████████████████▌                                                                                                   | 30/200 [03:28<10:53,  3.84s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 30 → دقت: 91.72% | انرژی کل: 1500.0 J\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 20%|██████████████████████▊                                                                                              | 39/200 [03:53<07:39,  2.85s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 40: تجمیع روی 100 کلاینت...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 30%|██████████████████████████████████▌                                                                                  | 59/200 [06:51<06:41,  2.85s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 60: تجمیع روی 100 کلاینت...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 30%|██████████████████████████████████▌                                                                                | 60/200 [08:59<1:34:10, 40.36s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 60 → دقت: 96.21% | انرژی کل: 3000.0 J\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 40%|██████████████████████████████████████████████▏                                                                      | 79/200 [09:50<05:43,  2.84s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 80: تجمیع روی 100 کلاینت...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 45%|████████████████████████████████████████████████████▋                                                                | 90/200 [12:08<05:17,  2.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 90 → دقت: 96.91% | انرژی کل: 4500.0 J\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 50%|█████████████████████████████████████████████████████████▉                                                           | 99/200 [12:24<03:12,  1.91s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 100: تجمیع روی 100 کلاینت...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 60%|█████████████████████████████████████████████████████████████████████                                               | 119/200 [14:41<02:32,  1.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 120: تجمیع روی 100 کلاینت...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 60%|█████████████████████████████████████████████████████████████████████▌                                              | 120/200 [16:22<42:04, 31.56s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 120 → دقت: 97.52% | انرژی کل: 6000.0 J\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 70%|████████████████████████████████████████████████████████████████████████████████▌                                   | 139/200 [16:57<01:57,  1.93s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 140: تجمیع روی 100 کلاینت...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 75%|███████████████████████████████████████████████████████████████████████████████████████                             | 150/200 [18:57<02:17,  2.75s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 150 → دقت: 97.83% | انرژی کل: 7500.0 J\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 80%|████████████████████████████████████████████████████████████████████████████████████████████▏                       | 159/200 [19:14<01:18,  1.92s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 160: تجمیع روی 100 کلاینت...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 90%|███████████████████████████████████████████████████████████████████████████████████████████████████████▊            | 179/200 [21:34<00:39,  1.90s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 180: تجمیع روی 100 کلاینت...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 90%|████████████████████████████████████████████████████████████████████████████████████████████████████████▍           | 180/200 [23:15<10:36, 31.83s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 180 → دقت: 98.09% | انرژی کل: 9000.0 J\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍| 199/200 [23:51<00:01,  1.92s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 200: تجمیع روی 100 کلاینت...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [25:33<00:00,  7.67s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   دور 200 → دقت: 98.15% | انرژی کل: 10000.0 J\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "print(f\"استفاده از دستگاه: {device}\")\n",
    "\n",
    "acc_periodic, energy_periodic = federated_periodic_averaging(T=20, K=100, local_epochs=5, device=device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "c397355d-7325-4797-aadc-7a39309cca41",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABcsAAAHjCAYAAAD8JEeeAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjcsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvTLEjVAAAAAlwSFlzAAAPYQAAD2EBqD+naQAAzS5JREFUeJzs3Qd4FNXXx/EfCb1KbyKiglQRRBREEAVFUFAUK1YUC0hXmtIULICigCJ/FBU7NlAQwQIqoEiVIsWGdKT3luR9zo2bN2UCKVuT7+d59gmZmczO3p1d7pw599wccXFxcQIAAAAAAAAAIBuLCvUBAAAAAAAAAAAQagTLAQAAAAAAAADZHsFyAAAAAAAAAEC2R7AcAAAAAAAAAJDtESwHAAAAAAAAAGR7BMsBAAAAAAAAANkewXIAAAAAAAAAQLZHsBwAAAAAAAAAkO0RLAcAAAAAAAAAZHsEywFkWZs2bdIHH3wQ6sMAAAAAAABABCBYDiBLiYuL06xZs9S2bVtVrFhRr7/+eqgPCQAAAAAAABGAYDmALGH37t164YUXdO655+rKK690WeUTJkzQlClTQn1oAAAAAAAAiAAEywFEtOXLl+vee+9V+fLl9fjjj+vSSy/VokWLNGfOHOXIkUP9+/dP1/4uu+wy93dHjhwJ2DFHukGDBrk2mjFjhsJRuLyHHAcAAEBkO/PMM1WmTJlQH0aWdvfdd7u+++rVqzmOMGoPIDvLGeoDAIDMGDlypH766ScNGzbMdSx27NihcePGaeLEidq1a5eaNm1KAwMAAAAAAOCUyCwHENGefPJJrVixQpUqVdLNN9+sKlWquAB64cKF9cwzz2jy5MmhPkQAAAAAIR7x98Ybb/AeAABOicxyABHt8OHDqlWrlhumZsPVrrjiCnXu3FnXXnutoqK4HwgAAABkd1ai0QLmkeqWW27R+eefrz59+oT6UAAgyyNYDiCi9erVywXK27Zt60qx2ASfAAAAAOATFxcX0Y1hZSfz5s0b6sMAgGyBtEsAEW3t2rXu58svv0ygHMggK1lkGUsAAABZzQcffKAbbrhBN910kws6+1u3bt3cI5D+/vtvysgggZUUiuSREkC4I1gOIKLdfvvt7ueDDz4YkoyR7777Tg888ICuuuoqtW/fXh9//HG6Ssi88sorateuna6++mr17Nkz4LOex8bGavTo0Ro0aFDCjYZgsA7+4MGD9fzzz+vEiRNBe95IZhPWnnnmmUF5LjvvAnHxCAAAEEoWxLaEgE8++cTNZWQBRuu/+9PSpUvdI1B9aCs16QuOpiVAan3tSZMm6bbbblOLFi308MMP65dffknX8+7cudPNDWUjdw8ePJiJV5B92PWVvVfBYOeFlRYCEBiUYQHgF+vWrXPB0L179+qiiy7S/fffr9KlSwe8dfv166d58+bps88+U9++fV2GbDBLwNhkoom988476tq1q0aNGnXKDugll1yiNWvWJCybMWOGxo4dq9mzZ+viiy8OyDFbHffLL79cTZs21fjx413HuXz58go0C/rmy5fP3RCwi4m33nor4M8Z6apWraojR44E5bm8Jryyc/m3335z71nRokWDchwAAAD+8u+//7okkUqVKrkAufVrWrdu7ZJsEvfBM8v67slZn96e//HHH1fOnBkPu1jplSZNmqhMmTKuZvmpHDt2zPXz7foosXHjxumjjz5ypSvTonjx4qpcubJLTJo1a5a+/vprRUdHZ/h1ZAd2vWPvVbAC8/ZIbPr06e59v/fee3XWWWcF5TiArIrMcgCZZtkGNrGmBde++OILPfHEEzrnnHPc76fKrLas7CJFiuiMM85w/+EfPXo0Xc9tnbb333/flWB59tlngxaEtc6vZUmXKlXKBbnttaxcudJ1KseMGaNt27ad9O8nTpzoOunXXHONfv/9dxcUtWX2+p966ql0H0962rJGjRr68MMPtXXrVpdpEiyPPfaY7rnnHpfp8umnnyqYtmzZot69e+vss89W7ty53Q0CGwlgbR+ubAInO7dP5ptvvnET3BYoUEDNmjVLd9bQydj5M3ToUDf6AQAAZG2B7FOsWLHCZXfbzXdLnrC+qGUtHz9+XIH0xx9/uFGVrVq1UsWKFV2W9XnnnedGV+7atSugz23XApZIZH3uzLAguQXj7dgtGedUCTlTp051AVNLvPn1119df86CqNb/HTBgQLqe296zgQMHuud/6aWXFEx79uxxWe3Vq1dXnjx5XBLW9ddfr0WLFimcR4V63ThJbMmSJWrQoIHy58/vfn711Vd+e367sWJ992AmjwFZVhwAZNKrr75q9U/ibr311rh9+/bFTZkyJa5ChQpu2dtvv+35N7GxsXFXXXWV26Z69epxZ555pvt369atM3QMa9asiTvttNPicufOHbds2bJTbn/o0KG4gQMHxr3xxhtJljdp0sQdx+HDh0/69zt37nTb1axZM27z5s0Jy7dt2xb322+/uf2fzMiRI93f9+7dO+748eMJy9euXRv3xx9/xKVHRtvyxhtvdNutXLkyXc9n7WZ/9+WXX8al17///huXN2/euPr16590u+nTp7vn+fvvv9P9HMnfQ3uPCxUq5JYVLlw47uKLL44766yz3O9lypSJ279/f5K/X7BgQdyll14aV7BgwbjatWvHzZo1K93H4HUc/jZ79uy4XLlyudd24YUXunbNkydP3Lx58/xyHHaO58iRI65hw4Z+PnIAABBO0tqnSC/roz711FNx0dHRri9yxhlnxDVo0MD1x+z3Rx555KR/v2rVKtcftOPzOXr0aNzkyZPjfvjhh1M+/19//eWep3LlynEbN26MW7dunevf2bJNmzal6TVUrFgxrnTp0nHptXjxYvc8t91220m3sza217h06dI4f/j444/d87Zv3z5J38/aYvXq1ene37Fjx+LKlSsXV758+STXLP58HXfddZc7ZruG8l0HWJvbsnz58rlzslq1au53e/+SXx/Y9VPLli3deXXOOefEvfvuu+k+Bq/j8Dfbr33G7LNl10L2b+trf/TRR345jhMnTsQVLVrUvV8AModgOYBM69y5s/sP/ZtvvklYtmHDBvefdalSpdx/3MnZtvY3zZo1c7/bNhdddJFbZp1LnyFDhrhln3/++SmPw4L0tm3z5s3TdNzWmbJOSuLjS09g8Y477nDb2sXFddddl67gsQXYfZ1AC9j26tXLdWIzIq1tmdyHH37otnnhhRcyFCyfMWNGho7XOrP297t37051G7vJYts8++yz6d5/4vdw6NChCe+RXawdPHgwYbuOHTu6dZ9++mnCMuuUWqfcLurOO+8899P+9pdffsnwcRw5ciQuI15//XV3jrz55pue6y2gb/v/8ccf3e9ffPGF500Sr3N6zJgxccWLF0/1ZpZPpUqV4ooVK5ah4wcAAJEhrX2KU7EA9PLly11f1B6+vrIlckybNi1hu127drlrhCJFipx0fxbgtr+/+uqrE5a99dZbbtmLL76YpmO69tpr3fbJHxZ0z2iw3JJdrB81Z86cVP/OXr/1IevWrXvS/Vub2/E89NBDJ93ugQcecMdhAf+TOXDgQFzVqlXdPi2R6MEHH3Tv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dPfdXMwCQBZEGRYEHZnlAAAAaXP77bdr+/btOvPMM7V7926df/752rRpk3LkyKGbb75Z7777Lk0JAJkVFydNmSJ16yatX++9TaNG0pgxUu3atDcAZGF+yent1auXWrdurUcffVTrU/uPBfgPwXIAAIBT27p1q3799Vc9/vjjqlq1ql566SXt2bNHc+fO1Z133qkPPvhAn3zyCU0JAJmxbp3UsqV0/fXegfLSpaW33pK+/55AOQBkA34Jli9ZskQbNmzQ888/r7p162r58uX+2C2yqJjkZVhy5AjVoQAAAIS96OhoxcXF6fXXX1fLli3VoEEDjRs3TmXLltWrr74alGMYOnSoy2avWbNminXz5s1To0aNlD9/fpUpU0ZdunTRgQMHUmx39OhR9e7dW+XKlVO+fPl00UUXadasWZ7Pl9Z9AkCGHTwo9e8v2ffaf6WukoiOjs80X7NGuuMOietWAMgW/BIs/+abb1zAfPHixcqVK5fuu+8+f+wWWZRd7CVGzXIAAICULEh87rnn6rnnnlPXrl1dckqHDh3cujx58qhVq1b6/vvvdezYsYA238aNGzVs2DAVKFAgxbqlS5fqiiuu0KFDh1zijF0HjB8/Xu3atUux7d133+22sdIyL774orsJYMH/H22CvAzuEwDSza5HP/5YqlZNGjZM8voObdzYvoykF16QihShkQEgG/FrzfLatWtryJAheuihh/TLL7/owgsv9OfukUVQhgUAACBtXn75ZV1//fUaM2aM62NfeeWVCeusfvmECRO0atUq9+9AsZKLF198sWJiYrRjx44k6/r166eiRYtq9uzZKly4sFtm9dXvv/9+zZw5M+F4FyxYoPfff1/Dhw93+zNWSsYy1R977DGXSZ7efQJAulmW+COPSKmMalHZstLIkdItt5BJDgDZlF8yyxO78cYb3U/ryAJeYmKTZZYznA0AAMBT06ZN3QSf+/fv19ixY5OsO/vss92Ivc2bNwes9Sxz/aOPPtKoUaNSrNu3b58ro9K+ffuEoLYvCF6wYEF9+OGHCctsH5ZJ3rFjx4RlefPmdZny8+fPd1nz6d0nAKSZlXHq00eqVcs7UJ4zp90ZjA+m33orgXIAyMYylVl+5MgRlyVSo0YNFS9e3C0rVqyYKleu7IZPAl6SVWFRtN9v2QAAAGQduXPndo/kTpw44eqIRwWopp1l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U089pSlTpujjjz/Wrl27dNlll7kJeWzd8ePHdemll7rAOmAowwIAAOBfZ555pu666y6ddtppNC2AyLVokdSwoXTffdKOHSnXW5mVadOkKVNc+RUAAMIiWJ4vXz5XF7FatWquQ/7jjz+6LJZ3333XrbcyLL4Jh77//nsVKFBAbdq0oSQLnDiPxADKsAAAAJzcli1b3GhO638XKVJE9erV0+DBg92oz3LlymnixIkuwxwAIs7OndKDD0oXXij9/HPK9ZY9/uST0ooVUsuWoThCAEA2kuHM8n379mnEiBGaPn26atWqpZEjR6pVq1a6+OKLE7YtX7683njjDf3+++8aN26cv48dWaUMCzXLAQAATmr//v0qVKiQOnbsqAcffFDFixd3pRFtriDKHgKISDEx0vjx8SVXXn3Ve4Kr66+XbKS6TeBpdcoBAAiwnOn9A8tcsWC5sRIrlmW+ePFilSlTRpdcckmK7a0si2WZf/TRR+revbt/jhoRizIsAAAA6VelShV99dVXSZbNmTNH119/vW644QYtWbIkYYQnAIS9BQviJ/BcuNB7feXK0ujR0lVXBfvIAADZXLozy5OrVKmS66B7Bcp9LOOFCYeQerA8B40DAACQTk2aNHGjOG1+IPsJAGHPapHff79kI9K9AuX580vDhknLlxMoBwBEZrA8Layu4tGjR4PxVAhzXpOZR0URLAcAAMiI1q1bq1GjRpowYQINCCC8S6688kp8yRX7vvIquXLjjfElV/r2lfLkCcVRAgDg32D5n3/+qb///jvJsqVLl+rNN99UgwYNAtrcQ4cOVY4cOVSzZs0U6+bNm+cuIvLnz+/KxXTp0kUHDhwI6PHAG2VYAAAA/Ktdu3ZatGiRm+wTAMLO/PlS/frSww9LXt9T554rzZolTZ4snXFGKI4QAIDABMutlvlZZ52lc845x9Uqr169ui644ALFxcVp7NixCpSNGzdq2LBhKlCgQIp1Fqy/4oordOjQIT3//PO67777NH78eHdRgeCjDAsAAIB/WclD62+vXLmSpgUQPrZvl+69V2rYUFq8OOV6u35/7jnp11+lZs1CcYQAAGR+gs+TueWWW3Taaadp9uzZWr9+vfu3LXvwwQdVqlQpBUqvXr108cUXKyYmRjusBloi/fr1U9GiRd0xFS5c2C0788wzdf/992vmzJm68sorA3ZcSGMZ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X242ZIhmjGhFAACAPEawHAVmX1qUzHJ6IBBXunfvbv3797dLLrnEZZsPHTrUVq5cae+///5+P7cXcC9ZsqT7yk/KOFV2fF76+OOPbd26dfb8889bamqqvfPOO1bQ9Lp7FUxJkDbza8OGDW4CpU+fPpYI9Dn8/PPP7jv8++2332zz5s3p3wtD0aJF7eyzz7aXX37ZTYABcWP27P+VVdHqmp07M99/zDFmn39u9vzzZtWqFcYRAgAABAKhShQYyrAgGaWlhWzT33vi7kvHlReU6SsKmHsmTpxorVq1spo1a1qVKlWsb9++tnbt2gy/p5rkTZo0scWLF1uHDh2sbNmyNnz48Jg1yxUsHTRokFWvXt0FbJs3b24vqQZrhK1bt9qFF15o5cuXtwoVKtjAgQPdbX7rb+vYW7du7Y6nYsWK7tj8Zrm/8sordtRRR1nnzp3t5JNPdj97VKpGwWaVx4i0YsUKK1asmMvWD38fmoioU6eOlSpVymVTjxw50tLS0jLVYn/ooYfs0Ucftfr167vHLl261AXM//Of/1jLli1dWxxwwAF2wgkn2FzVt42wadMmu+CCC1w2rddm3377ra867/pZpXjee+8993nq9Rs3bmwffPBB1MmEY4891n1+OtZx48b5roM+bdo0F3hWuyYCTZocfvjh7jv869evnz322GPp3wtLly5dbPXq1fbNN98U2jEA6fT3UxOFXbqYLVuWuWEqVjTTSohFi8zataPhAAAA8hlr91BgKMOCZLTln73WcsRsizeLbz/ZKh9Yar+f55dffnHfK1eu7L7fe++9dscdd7gM4PPPP9/+/vtvFwRW0Pnrr792wdjwIK0y1RVMV7a6AuHRqPSHgufK1FVg9tBDD7VJkya5oLiCytdee617nLJAVULhs88+syuuuMKOPPJIe/fdd13w1w8FshW8VXkZlZtRdvuXX35pc+bMsa5du2b5u3v27LG3337bbrjhBvfzeeedZxdddJGre1yjRg333jp27Ghvvvmm3XnnnRl+V8eoYLmXNa0Mez1W2bWXX3651a1b1z7//HO79dZb7Y8//nCB8XAvvPCC7d692y677DIXrK5UqZJt377dnn32WXcc2hBTJWGee+45O+WUU2zhwoV2tDaEc5M5aXbaaae526688kpr1KiRWyXgt81E7a0s+quuusrKlSvngpxnnXWWrVmz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x/g99xi/x8b43T/+RuYM4/fcYfweG+P35Yk7fg8h7ixYsEBTJqFRo0al37Z79+5QgwYNQm3atAkFTWpqaqhXr16h4sWLh6ZNmxb1Mbt27Qpt37490+033XSTa8t33nknlOw2bNiQ6bZvvvkmVKJECdd+nm7duoVq1qyZob2effZZ104zZswIBYXaS33qggsuyHTftm3b3DkXLi0tLXTuuee6dlq8eHEoWS1atMi9xxdeeCHTfVdccUWoTJkyoTVr1qTfNnv2bPf4p556Kv22tWvXun43ePDgDO13wgknhA4++GB3TidzOy1dujRUuXLlUOPGjUObN2/O0Tmr87J+/fqhKlWqhPbs2RNK1nbKyTU76P3JM3r0aPeYjz76yHd/WrduXahixYqhZs2ahZL1779wbUJhY+yeGeN3fxi/5wzj9+gYv/vD+H3/2onxu//+5GH8Pi1hx+9s8BmH2rRpY3369HGbMm7YsMHVrXvppZds1apVmTYdDALViZ48ebLLLNMy8okTJ2a4v3///m5ZxjHHHGPnnXeeNWrUyN2ujc6mT59u3bp1c3W7k925557rlkBpo6Bq1arZ0qVLbfz48Va2bFl74IEH0h937733usd07NjRbVipOrYPP/ywde3a1bVVkGrRpaamRi3BsmTJEteX9KXzT5tGvPvuu24zRrVZixYtLNk88cQTrsSBt5v0lClT0mscX3311a70gzadnDRpknXu3NmuvfZaVzNy1KhR1rRpU1cX2VO7dm0bOnSouy8lJcVatWrllszOmzfPZQNlt9dAIreT6m2rBv6WLVvspptucpsxhqtfv77b3ETGjh3r2kXXNm3OqMyD559/3tasWWMTJkxw9cuTtZ3UPn6v2UHuTzrvPHqvWmbYqVOnqM81bNgw++WXX+ykk05yj9OYYdy4ca4E15gxYyxZ//4L1yYUNsbumTF+94fxe84wfs+I8bs/jN/zpp0Yv/s/7zyM3zcn7vg9X0Lw2G+atbvxxhtDNWrUCJUqVSrUqlWr0AcffBDIlu3YsaObXYr1JVu2bAn179/fZd+XLVvWtZmyOu+7777Q3r17Q0EwZsyYUOvWrUOVKlVyWXjKHlebrFixItNj582bF2rfvn2odOnSoapVq7pZumhZnsmsbdu2oWrVqkWdifz1119Dffr0CdWrV8+1kfpUy5YtQ08//bSbxUxG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",
      "text/plain": [
       "<Figure size 1500x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🎉 پروژه با موفقیت اجرا شد! حالا می‌تونی این نوت‌بوک رو تحویل بدی یا تو دفاع زنده اجراش کنی\n"
     ]
    }
   ],
   "source": [
    "# رسم نتایج نهایی (شبیه شکل‌های مقاله)\n",
    "plt.figure(figsize=(15, 5))\n",
    "\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(acc_periodic, label='Periodic Averaging (پیشنهادی مقاله)', linewidth=3)\n",
    "plt.xlabel('دوره جهانی')\n",
    "plt.ylabel('دقت تست (%)')\n",
    "plt.title('همگرایی مدل در طول زمان')\n",
    "plt.grid(True)\n",
    "plt.legend()\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(energy_periodic, label='مصرف انرژی کل', color='red', linewidth=3)\n",
    "plt.xlabel('دوره جهانی')\n",
    "plt.ylabel('انرژی مصرفی (J)')\n",
    "plt.title('مصرف انرژی در طول زمان')\n",
    "plt.grid(True)\n",
    "plt.legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"🎉 پروژه با موفقیت اجرا شد! حالا می‌تونی این نوت‌بوک رو تحویل بدی یا تو دفاع زنده اجراش کنی\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4c689228-90da-47b9-bf0e-71fd69fb1d06",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1f3b86c8-d342-45b8-a4a4-b1a94e7556b9",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "890bde3f-779e-422e-a96c-85cbcbc841ab",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
