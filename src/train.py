#!/usr/bin/env python3
"""
ATP Tennis Match Predictor - Training & Complete Analysis
Basato su Completo_1.ipynb con analisi estesa per:
- Metriche globali
- Performance per superficie
- Performance PER GIOCATORE (prevedibilità)
- Strategie betting basate su confidence, upset, quote
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, 
                             classification_report, brier_score_loss, log_loss)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
import joblib
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)

# Configurazione
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("ATP TENNIS MATCH PREDICTOR - TRAINING & COMPLETE ANALYSIS")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ==============================================================================
# 1) CARICAMENTO DATI
# ==============================================================================
print("1) Caricamento dati...")
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'atp_tennis_cleaned.csv'), parse_dates=['Date'])
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
start, end = pd.to_datetime('2010-01-01'), pd.to_datetime('2025-11-23')
df = df[(df['Date'] >= start) & (df['Date'] <= end)].copy().sort_values('Date')
print(f"   Partite caricate: {len(df):,}")
print(f"   Periodo: {df['Date'].min().date()} - {df['Date'].max().date()}")

# ==============================================================================
# 2) FEATURE ENGINEERING
# ==============================================================================
print("\n2) Feature Engineering...")

# Target
df['p1_win'] = (df['Winner'] == df['Player_1']).astype(int)
df['p2_win'] = (df['Winner'] == df['Player_2']).astype(int)

# Statistiche mobili per giocatore
print("   Calcolo statistiche mobili per giocatore...")
player_stats = []
all_players = set(df['Player_1']).union(set(df['Player_2']))
for player in tqdm(all_players, desc="   Elaborazione giocatori"):
    player_matches = pd.concat([
        df[df['Player_1'] == player][['Date', 'p1_win']].rename(columns={'p1_win': 'win'}),
        df[df['Player_2'] == player][['Date', 'p2_win']].rename(columns={'p2_win': 'win'})
    ]).sort_values('Date')
    
    player_matches['win_rate'] = player_matches['win'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean().shift(1)
    )
    player_matches['streak'] = player_matches['win'].groupby(
        (player_matches['win'] != player_matches['win'].shift()).cumsum()
    ).cumcount() + 1
    player_matches['streak'] = player_matches['streak'] * player_matches['win']
    
    player_stats.append(player_matches[['Date', 'win_rate', 'streak']].assign(player=player))

player_stats = pd.concat(player_stats)

# Merge statistiche
print("   Merge statistiche...")
def merge_player_stats(df_row, stats_df, player_col, date_col):
    stats = stats_df[
        (stats_df['player'] == df_row[player_col]) & 
        (stats_df['Date'] < df_row['Date'])
    ].sort_values('Date', ascending=False)
    
    if stats.empty:
        return 0.5, 0
    else:
        return stats.iloc[0]['win_rate'], stats.iloc[0]['streak']

df[['p1_win_rate', 'p1_streak']] = df.apply(
    lambda x: merge_player_stats(x, player_stats, 'Player_1', 'Date'), 
    axis=1, result_type='expand'
)
df[['p2_win_rate', 'p2_streak']] = df.apply(
    lambda x: merge_player_stats(x, player_stats, 'Player_2', 'Date'), 
    axis=1, result_type='expand'
)

# Head-to-Head
print("   Calcolo Head-to-Head...")
h2h = defaultdict(lambda: {'p1_wins': 0, 'matches': 0})
h2h_features = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="   Calcolo H2H"):
    p1, p2 = row['Player_1'], row['Player_2']
    key = (p1, p2)
    
    if key not in h2h:
        h2h[key] = {'p1_wins': 0, 'matches': 0}
    
    if h2h[key]['matches'] > 0:
        h2h_win_rate = h2h[key]['p1_wins'] / h2h[key]['matches']
    else:
        h2h_win_rate = 0.5
    
    h2h_features.append({
        'h2h_win_rate': h2h_win_rate,
        'h2h_matches': h2h[key]['matches']
    })
    
    if row['Winner'] == p1:
        h2h[key]['p1_wins'] += 1
    h2h[key]['matches'] += 1

df = pd.concat([df, pd.DataFrame(h2h_features)], axis=1)

# Feature invarianti
print("   Creazione feature invarianti...")
df['rank_diff'] = df['Rank_1'] - df['Rank_2']
df['odds_ratio'] = df['Odd_1'] / df['Odd_2']
df['win_rate_diff'] = df['p1_win_rate'] - df['p2_win_rate']
df['streak_diff'] = df['p1_streak'] - df['p2_streak']

df['min_rank'] = np.minimum(df['Rank_1'], df['Rank_2'])
df['max_rank'] = np.maximum(df['Rank_1'], df['Rank_2'])
df['min_odds'] = np.minimum(df['Odd_1'], df['Odd_2'])
df['max_odds'] = np.maximum(df['Odd_1'], df['Odd_2'])
df['max_win_rate'] = np.maximum(df['p1_win_rate'], df['p2_win_rate'])
df['min_win_rate'] = np.minimum(df['p1_win_rate'], df['p2_win_rate'])
df['max_streak'] = np.maximum(df['p1_streak'], df['p2_streak'])
df['min_streak'] = np.minimum(df['p1_streak'], df['p2_streak'])

# ==============================================================================
# 3) SPLIT DATI
# ==============================================================================
print("\n3) Split dei dati...")
n = len(df)
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)
t0, t1 = int(0.7*n), int(0.85*n)
train_i, val_i, test_i = idx[:t0], idx[t0:t1], idx[t1:]

print(f"   Training: {len(train_i):,} partite")
print(f"   Validation: {len(val_i):,} partite")
print(f"   Test: {len(test_i):,} partite")

# Definizione colonne
num_cols = ['rank_diff', 'odds_ratio', 'win_rate_diff', 'streak_diff',
            'min_rank', 'max_rank', 'min_odds', 'max_odds',
            'max_win_rate', 'min_win_rate', 'max_streak', 'min_streak',
            'h2h_win_rate', 'h2h_matches']
cat_cols = ['Surface', 'Tournament']

# ==============================================================================
# 4) PREPARAZIONE DATI
# ==============================================================================
print("\n4) Preparazione dati per il modello...")
X_num = df[num_cols].fillna(0).values

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat = encoder.fit_transform(df[['Surface','Tournament']].fillna('NA'))

scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_num[train_i])
X_num = scaler.transform(X_num)

X = np.hstack([X_num, X_cat])
y = df['p1_win'].fillna(0).astype(int).values

print(f"   Feature numeriche: {len(num_cols)}")
print(f"   Feature categoriche (one-hot): {X_cat.shape[1]}")
print(f"   Feature totali: {X.shape[1]}")

# ==============================================================================
# 5) DATASET E DATALOADER
# ==============================================================================
class TDataset(Dataset):
    def __init__(self, X, y, i):
        self.X = torch.tensor(X[i], dtype=torch.float32)
        self.y = torch.tensor(y[i], dtype=torch.float32).unsqueeze(1)
        
    def __len__(self): 
        return len(self.y)
    
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]

train_ds = TDataset(X, y, train_i)
val_ds = TDataset(X, y, val_i)
test_ds = TDataset(X, y, test_i)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)
test_loader = DataLoader(test_ds, batch_size=128)

# ==============================================================================
# 6) MODELLO
# ==============================================================================
class EnhancedMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), 
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4929775),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 128), 
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4220651),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1701431),
            
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(self.bn1(x))

# ==============================================================================
# 7) TRAINING
# ==============================================================================
print("\n5) Inizializzazione modello e training...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

model = EnhancedMLP(X.shape[1]).to(device)

pos_samples = np.sum(y[train_i])
neg_samples = len(y[train_i]) - pos_samples
pos_weight = ((neg_samples / pos_samples) + 5.05) if pos_samples > 0 else 1.0

print(f"   Campioni positivi (P1 wins): {pos_samples:,}")
print(f"   Campioni negativi (P2 wins): {neg_samples:,}")
print(f"   Peso positivo: {pos_weight:.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
optimizer = optim.Adam(model.parameters(), lr=9.00064e-05, weight_decay=8.196026e-05)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.28)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    auc = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    auc = roc_auc_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, np.array(all_preds) > 0.5)
    return avg_loss, auc, accuracy, np.array(all_preds).flatten(), np.array(all_labels).flatten()

# Training loop
print("\n   Inizio training...")
n_epochs = 100
best_val_auc = 0
patience = 30
patience_counter = 0
best_model = None

train_losses, val_losses = [], []
train_aucs, val_aucs = [], []

for epoch in range(n_epochs):
    train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_aucs.append(train_auc)
    
    val_loss, val_auc, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_aucs.append(val_auc)
    
    scheduler.step(val_auc)
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        best_model = model.state_dict().copy()
        torch.save(best_model, f'{OUTPUT_DIR}/best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1}: Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, Val Acc={val_acc:.4f}")

# Carica miglior modello
model.load_state_dict(torch.load(f'{OUTPUT_DIR}/best_model.pt'))

# ==============================================================================
# 8) VALUTAZIONE FINALE
# ==============================================================================
print("\n" + "="*80)
print("6) VALUTAZIONE FINALE SUL TEST SET")
print("="*80)

test_loss, test_auc, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

# Metriche principali
test_preds_binary = (test_preds > 0.5).astype(int)
test_labels_int = test_labels.astype(int)

print(f"\n   AUC-ROC: {test_auc:.4f}")
print(f"   Accuracy: {test_acc:.4f}")
print(f"   Precision: {precision_score(test_labels_int, test_preds_binary):.4f}")
print(f"   Recall: {recall_score(test_labels_int, test_preds_binary):.4f}")
print(f"   F1-Score: {f1_score(test_labels_int, test_preds_binary):.4f}")
print(f"   Brier Score: {brier_score_loss(test_labels_int, test_preds):.4f}")
print(f"   Log Loss: {log_loss(test_labels_int, test_preds):.4f}")

# Confusion Matrix
cm = confusion_matrix(test_labels_int, test_preds_binary)
print(f"\n   Confusion Matrix:")
print(f"   TN={cm[0,0]:,}, FP={cm[0,1]:,}")
print(f"   FN={cm[1,0]:,}, TP={cm[1,1]:,}")

# ==============================================================================
# 9) ANALISI PER SUPERFICIE
# ==============================================================================
print("\n" + "="*80)
print("7) ANALISI PER SUPERFICIE")
print("="*80)

# Crea dataframe con predizioni per il test set
df_test = df.iloc[test_i].copy()
df_test['pred_prob'] = test_preds
df_test['pred_class'] = test_preds_binary
df_test['correct'] = (df_test['pred_class'] == df_test['p1_win']).astype(int)

surface_analysis = []
for surface in df_test['Surface'].unique():
    surface_df = df_test[df_test['Surface'] == surface]
    if len(surface_df) >= 50:  # Minimo 50 partite
        acc = surface_df['correct'].mean()
        n_matches = len(surface_df)
        avg_odds = surface_df['min_odds'].mean()
        
        # Upset detection
        favorites_win = (surface_df['Odd_1'] < surface_df['Odd_2']) == (surface_df['p1_win'] == 1)
        upset_rate = 1 - favorites_win.mean()
        
        surface_analysis.append({
            'Surface': surface,
            'Matches': n_matches,
            'Accuracy': acc,
            'Upset_Rate': upset_rate,
            'Avg_Min_Odds': avg_odds
        })

surface_df = pd.DataFrame(surface_analysis).sort_values('Accuracy', ascending=False)
print("\n   Performance per Superficie:")
print(surface_df.to_string(index=False))
surface_df.to_csv(f'{OUTPUT_DIR}/surface_analysis.csv', index=False)

# ==============================================================================
# 10) ANALISI PER GIOCATORE - LA PIÙ IMPORTANTE
# ==============================================================================
print("\n" + "="*80)
print("8) ANALISI PER GIOCATORE (PREVEDIBILITÀ)")
print("="*80)

# Calcola statistiche per ogni giocatore nel test set
player_stats_analysis = defaultdict(lambda: {
    'matches': 0, 'correct': 0, 'as_favorite': 0, 'as_underdog': 0,
    'favorite_correct': 0, 'underdog_correct': 0, 'upset_caused': 0,
    'upset_suffered': 0, 'avg_confidence': [], 'odds_when_favorite': [],
    'odds_when_underdog': []
})

for _, row in df_test.iterrows():
    p1, p2 = row['Player_1'], row['Player_2']
    p1_win = row['p1_win']
    correct = row['correct']
    pred_prob = row['pred_prob']
    
    # Determina favorito
    p1_is_favorite = row['Odd_1'] < row['Odd_2']
    
    # Statistiche P1
    player_stats_analysis[p1]['matches'] += 1
    player_stats_analysis[p1]['correct'] += correct
    player_stats_analysis[p1]['avg_confidence'].append(pred_prob if p1_win else 1-pred_prob)
    
    if p1_is_favorite:
        player_stats_analysis[p1]['as_favorite'] += 1
        player_stats_analysis[p1]['odds_when_favorite'].append(row['Odd_1'])
        if correct:
            player_stats_analysis[p1]['favorite_correct'] += 1
        if p1_win == 0:  # Upset (favorito ha perso)
            player_stats_analysis[p1]['upset_suffered'] += 1
    else:
        player_stats_analysis[p1]['as_underdog'] += 1
        player_stats_analysis[p1]['odds_when_underdog'].append(row['Odd_1'])
        if correct:
            player_stats_analysis[p1]['underdog_correct'] += 1
        if p1_win == 1:  # Upset causato (underdog ha vinto)
            player_stats_analysis[p1]['upset_caused'] += 1
    
    # Statistiche P2
    player_stats_analysis[p2]['matches'] += 1
    player_stats_analysis[p2]['correct'] += correct
    player_stats_analysis[p2]['avg_confidence'].append(1-pred_prob if p1_win else pred_prob)
    
    if not p1_is_favorite:
        player_stats_analysis[p2]['as_favorite'] += 1
        player_stats_analysis[p2]['odds_when_favorite'].append(row['Odd_2'])
        if correct:
            player_stats_analysis[p2]['favorite_correct'] += 1
        if p1_win == 1:  # Upset (p2 favorito ha perso)
            player_stats_analysis[p2]['upset_suffered'] += 1
    else:
        player_stats_analysis[p2]['as_underdog'] += 1
        player_stats_analysis[p2]['odds_when_underdog'].append(row['Odd_2'])
        if correct:
            player_stats_analysis[p2]['underdog_correct'] += 1
        if p1_win == 0:  # Upset causato
            player_stats_analysis[p2]['upset_caused'] += 1

# Converti in DataFrame
player_analysis = []
for player, stats in player_stats_analysis.items():
    # Salta giocatori senza nome valido
    if pd.isna(player) or player == '' or str(player).strip() == '':
        continue
    if stats['matches'] >= 10:  # Minimo 10 partite
        accuracy = stats['correct'] / stats['matches']
        
        fav_acc = stats['favorite_correct'] / stats['as_favorite'] if stats['as_favorite'] > 0 else 0
        und_acc = stats['underdog_correct'] / stats['as_underdog'] if stats['as_underdog'] > 0 else 0
        
        upset_caused_rate = stats['upset_caused'] / stats['as_underdog'] if stats['as_underdog'] > 0 else 0
        upset_suffered_rate = stats['upset_suffered'] / stats['as_favorite'] if stats['as_favorite'] > 0 else 0
        
        avg_conf = np.mean(stats['avg_confidence']) if stats['avg_confidence'] else 0.5
        
        avg_odds_fav = np.mean(stats['odds_when_favorite']) if stats['odds_when_favorite'] else 0
        avg_odds_und = np.mean(stats['odds_when_underdog']) if stats['odds_when_underdog'] else 0
        
        player_analysis.append({
            'Player': player,
            'Matches': stats['matches'],
            'Accuracy': accuracy,
            'As_Favorite': stats['as_favorite'],
            'As_Underdog': stats['as_underdog'],
            'Fav_Accuracy': fav_acc,
            'Und_Accuracy': und_acc,
            'Upset_Caused_Rate': upset_caused_rate,
            'Upset_Suffered_Rate': upset_suffered_rate,
            'Avg_Confidence': avg_conf,
            'Avg_Odds_Favorite': avg_odds_fav,
            'Avg_Odds_Underdog': avg_odds_und,
            'Predictability_Score': accuracy * (1 - abs(upset_caused_rate - upset_suffered_rate))
        })

player_df = pd.DataFrame(player_analysis)

# Filtra giocatori con nomi validi
player_df = player_df[player_df['Player'].notna() & (player_df['Player'] != '')].copy()

# Giocatori più FACILI da predire
print("\n   TOP 20 GIOCATORI PIÙ FACILI DA PREDIRE (Alta Accuracy):")
easy_players = player_df[player_df['Matches'] >= 20].nlargest(20, 'Accuracy')
print(easy_players[['Player', 'Matches', 'Accuracy', 'Fav_Accuracy', 'Und_Accuracy', 'Upset_Suffered_Rate']].to_string(index=False))

# Giocatori più DIFFICILI da predire
print("\n   TOP 20 GIOCATORI PIÙ DIFFICILI DA PREDIRE (Bassa Accuracy):")
hard_players = player_df[player_df['Matches'] >= 20].nsmallest(20, 'Accuracy')
print(hard_players[['Player', 'Matches', 'Accuracy', 'Upset_Caused_Rate', 'Upset_Suffered_Rate']].to_string(index=False))

# Giocatori che CAUSANO più upset
print("\n   TOP 20 GIOCATORI CHE CAUSANO PIÙ UPSET (Giant Killers):")
upset_makers = player_df[(player_df['As_Underdog'] >= 10)].nlargest(20, 'Upset_Caused_Rate')
print(upset_makers[['Player', 'As_Underdog', 'Upset_Caused_Rate', 'Avg_Odds_Underdog']].to_string(index=False))

# Giocatori che SUBISCONO più upset
print("\n   TOP 20 GIOCATORI CHE SUBISCONO PIÙ UPSET (Unreliable Favorites):")
upset_victims = player_df[(player_df['As_Favorite'] >= 10)].nlargest(20, 'Upset_Suffered_Rate')
print(upset_victims[['Player', 'As_Favorite', 'Upset_Suffered_Rate', 'Avg_Odds_Favorite']].to_string(index=False))

# Salva tutto
player_df.to_csv(f'{OUTPUT_DIR}/player_analysis.csv', index=False)
print(f"\n   Analisi giocatori salvata in: {OUTPUT_DIR}/player_analysis.csv")

# ==============================================================================
# 11) ANALISI STRATEGIE BETTING
# ==============================================================================
print("\n" + "="*80)
print("9) ANALISI STRATEGIE BETTING")
print("="*80)

# Calcola confidence come distanza da 0.5
df_test['confidence'] = np.abs(df_test['pred_prob'] - 0.5) * 2  # 0-1 scale

# Upset signal: quando modello predice underdog
df_test['p1_is_favorite'] = df_test['Odd_1'] < df_test['Odd_2']
df_test['model_predicts_p1'] = df_test['pred_prob'] > 0.5
df_test['upset_signal'] = (
    ((df_test['model_predicts_p1']) & (~df_test['p1_is_favorite'])) |
    ((~df_test['model_predicts_p1']) & (df_test['p1_is_favorite']))
)

# Calcola ROI per diverse strategie
def calculate_strategy_roi(df_subset, bet_on_model=True):
    """Calcola ROI scommettendo sulla predizione del modello"""
    if len(df_subset) == 0:
        return 0, 0, 0
    
    # Filtra righe con odds validi
    valid_subset = df_subset[(df_subset['Odd_1'] > 0) & (df_subset['Odd_2'] > 0)].copy()
    if len(valid_subset) == 0:
        return 0, 0, 0
    
    total_bets = len(valid_subset)
    total_stake = total_bets  # 1 unità per scommessa
    total_returns = 0
    wins = 0
    
    for _, row in valid_subset.iterrows():
        if bet_on_model:
            # Scommetti sul giocatore predetto dal modello
            bet_on_p1 = row['pred_prob'] > 0.5
        else:
            # Scommetti sempre sul favorito
            bet_on_p1 = row['Odd_1'] < row['Odd_2']
        
        actual_p1_wins = row['p1_win'] == 1
        
        if bet_on_p1:
            odds = row['Odd_1']
            if actual_p1_wins:
                total_returns += odds
                wins += 1
        else:
            odds = row['Odd_2']
            if not actual_p1_wins:
                total_returns += odds
                wins += 1
    
    roi = ((total_returns - total_stake) / total_stake) * 100 if total_stake > 0 else 0
    accuracy = wins / total_bets if total_bets > 0 else 0
    return roi, accuracy, total_bets

# Strategia 1: Tutte le scommesse
roi_all, acc_all, n_all = calculate_strategy_roi(df_test)
print(f"\n   STRATEGIA 1 - Scommetti su tutte le predizioni:")
print(f"   Scommesse: {n_all}, Accuracy: {acc_all:.2%}, ROI: {roi_all:.2f}%")

# Strategia 2: Solo alta confidence
strategies = []
for conf_threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    high_conf = df_test[df_test['confidence'] >= conf_threshold]
    roi, acc, n = calculate_strategy_roi(high_conf)
    strategies.append({
        'Strategy': f'Confidence >= {conf_threshold:.0%}',
        'Bets': n,
        'Accuracy': acc,
        'ROI': roi
    })

# Strategia 3: Solo upset signals
upset_bets = df_test[df_test['upset_signal'] == True]
roi_upset, acc_upset, n_upset = calculate_strategy_roi(upset_bets)
strategies.append({
    'Strategy': 'Solo Upset Signals',
    'Bets': n_upset,
    'Accuracy': acc_upset,
    'ROI': roi_upset
})

# Strategia 4: Upset + alta confidence
for conf_threshold in [0.3, 0.4, 0.5]:
    upset_high_conf = df_test[(df_test['upset_signal'] == True) & (df_test['confidence'] >= conf_threshold)]
    roi, acc, n = calculate_strategy_roi(upset_high_conf)
    strategies.append({
        'Strategy': f'Upset + Conf >= {conf_threshold:.0%}',
        'Bets': n,
        'Accuracy': acc,
        'ROI': roi
    })

# Strategia 5: Solo favoriti forti (quote basse)
for odds_threshold in [1.3, 1.5, 1.7]:
    strong_fav = df_test[df_test['min_odds'] <= odds_threshold]
    roi, acc, n = calculate_strategy_roi(strong_fav, bet_on_model=False)
    strategies.append({
        'Strategy': f'Solo favoriti odds <= {odds_threshold}',
        'Bets': n,
        'Accuracy': acc,
        'ROI': roi
    })

# Strategia 6: Value bets (modello concorda con underdog)
value_bets = df_test[
    (df_test['upset_signal'] == True) & 
    (df_test['confidence'] >= 0.3) &
    (df_test['max_odds'] >= 2.0)
]
roi_value, acc_value, n_value = calculate_strategy_roi(value_bets)
strategies.append({
    'Strategy': 'Value Bets (Upset+Conf+Odds>=2.0)',
    'Bets': n_value,
    'Accuracy': acc_value,
    'ROI': roi_value
})

# Strategia 7: Combinata ottimale
combined = df_test[
    ((df_test['confidence'] >= 0.6) & (~df_test['upset_signal'])) |  # High conf favoriti
    ((df_test['confidence'] >= 0.4) & (df_test['upset_signal']) & (df_test['max_odds'] >= 2.5))  # Upset con value
]
roi_combined, acc_combined, n_combined = calculate_strategy_roi(combined)
strategies.append({
    'Strategy': 'Combinata Ottimale',
    'Bets': n_combined,
    'Accuracy': acc_combined,
    'ROI': roi_combined
})

# Mostra risultati base
strat_df = pd.DataFrame(strategies).sort_values('ROI', ascending=False)
print("\n   CONFRONTO STRATEGIE BASE (ordinate per ROI):")
print(strat_df.to_string(index=False))

# ==============================================================================
# 9B) STRATEGIE AVANZATE: UPSET SIGNAL + PLAYER RELIABILITY + CONFIDENCE
# ==============================================================================
print("\n" + "="*80)
print("9B) STRATEGIE AVANZATE CON PLAYER RELIABILITY")
print("="*80)

# 1. Calcola UPSET SIGNAL avanzato (come in Completo_1.ipynb)
def calcola_segnali_upset_avanzato(row):
    """
    Calcola segnali che indicano un upset potenziale legittimo
    Ritorna un punteggio 0-1 dove 1 = segnali fortissimi di upset
    (Implementazione da Completo_1.ipynb)
    """
    segnali = 0
    
    # 1. Form recente molto diversa
    form_diff = row.get('p1_win_rate', 0.5) - row.get('p2_win_rate', 0.5)
    if form_diff > 0.2:  # P1 in forma molto migliore
        segnali += 0.3
    elif form_diff > 0.1:
        segnali += 0.15
    
    # 2. Streak positivo per underdog
    if row.get('p1_streak', 0) > 3:  # P1 su buona striscia
        segnali += 0.2
    
    # 3. H2H favorevole
    if row.get('h2h_win_rate', 0.5) > 0.6 and row.get('h2h_matches', 0) >= 3:
        segnali += 0.25
    
    # 4. Odds vs ranking inconsistency
    rank_1 = max(row.get('Rank_1', 100), 1)
    rank_2 = max(row.get('Rank_2', 100), 1)
    odd_1 = max(row.get('Odd_1', 1.5), 1.01)
    odd_2 = max(row.get('Odd_2', 1.5), 1.01)
    
    rank_ratio = rank_2 / rank_1
    odds_ratio = odd_2 / odd_1
    
    # Se market è più ottimista su P1 rispetto ai ranking
    if odds_ratio < rank_ratio * 0.8:
        segnali += 0.15
    
    return min(segnali, 1.0)

# Applica segnali upset avanzati
print("\n   Calcolo segnali upset avanzati (come Completo_1)...")
df_test['upset_signal_score'] = df_test.apply(calcola_segnali_upset_avanzato, axis=1)

# 2. Crea PLAYER RELIABILITY INDEX basato su accuracy storica
print("   Creazione indice affidabilità giocatori...")

# Usa player_stats_analysis già calcolato per ottenere reliability score
player_reliability = {}
for player, stats in player_stats_analysis.items():
    if pd.isna(player) or player == '' or str(player).strip() == '':
        continue
    if stats['matches'] >= 10:
        # Reliability = accuracy storica (quanto è prevedibile)
        accuracy = stats['correct'] / stats['matches'] if stats['matches'] > 0 else 0.5
        player_reliability[player] = accuracy

# Media reliability per giocatori sconosciuti
avg_reliability = np.mean(list(player_reliability.values())) if player_reliability else 0.5

# Funzione per calcolare reliability combinata della partita
def get_match_reliability(row):
    """Calcola affidabilità della partita basata sui due giocatori"""
    p1_rel = player_reliability.get(row['Player_1'], avg_reliability)
    p2_rel = player_reliability.get(row['Player_2'], avg_reliability)
    # Media pesata: più partite = più peso
    return (p1_rel + p2_rel) / 2

df_test['match_reliability'] = df_test.apply(get_match_reliability, axis=1)

# 3. Crea COMPOSITE SCORE: Confidence + Reliability - UpsetRisk
# Composite = alta confidence + alta affidabilità giocatori = scommessa sicura
# UpsetScore alto = più rischio
df_test['composite_score'] = (
    df_test['confidence'] * 0.4 +  # Peso confidence modello
    df_test['match_reliability'] * 0.4 +  # Peso affidabilità giocatori
    (1 - df_test['upset_signal_score']) * 0.2  # Inverso del rischio upset
)

print(f"\n   Statistiche indicatori:")
print(f"   - Upset Signal Score: mean={df_test['upset_signal_score'].mean():.3f}, max={df_test['upset_signal_score'].max():.3f}")
print(f"   - Match Reliability: mean={df_test['match_reliability'].mean():.3f}, std={df_test['match_reliability'].std():.3f}")
print(f"   - Composite Score: mean={df_test['composite_score'].mean():.3f}")

# 4. STRATEGIE COMBINATE AVANZATE
advanced_strategies = []

# Strategia A: Alta affidabilità + alta confidence
for rel_thresh in [0.55, 0.60, 0.65]:
    for conf_thresh in [0.5, 0.6, 0.7]:
        subset = df_test[
            (df_test['match_reliability'] >= rel_thresh) & 
            (df_test['confidence'] >= conf_thresh)
        ]
        roi, acc, n = calculate_strategy_roi(subset)
        if n >= 50:  # Almeno 50 scommesse
            advanced_strategies.append({
                'Strategy': f'Reliability>={rel_thresh:.0%} + Conf>={conf_thresh:.0%}',
                'Bets': n,
                'Accuracy': acc,
                'ROI': roi,
                'Type': 'Safe'
            })

# Strategia B: Basso rischio upset + alta confidence
for upset_thresh in [0.1, 0.2, 0.3]:
    for conf_thresh in [0.5, 0.6]:
        subset = df_test[
            (df_test['upset_signal_score'] <= upset_thresh) & 
            (df_test['confidence'] >= conf_thresh) &
            (~df_test['upset_signal'])  # Modello concorda con favorito
        ]
        roi, acc, n = calculate_strategy_roi(subset)
        if n >= 50:
            advanced_strategies.append({
                'Strategy': f'LowUpsetRisk<={upset_thresh:.0%} + Conf>={conf_thresh:.0%}',
                'Bets': n,
                'Accuracy': acc,
                'ROI': roi,
                'Type': 'Conservative'
            })

# Strategia C: Composite Score alto
for comp_thresh in [0.5, 0.55, 0.6, 0.65]:
    subset = df_test[df_test['composite_score'] >= comp_thresh]
    roi, acc, n = calculate_strategy_roi(subset)
    if n >= 50:
        advanced_strategies.append({
            'Strategy': f'CompositeScore>={comp_thresh:.0%}',
            'Bets': n,
            'Accuracy': acc,
            'ROI': roi,
            'Type': 'Composite'
        })

# Strategia D: Value Upset - Alto upset signal + alta reliability underdog
# Scommetti sull'underdog quando ha segnali forti E è affidabile
subset = df_test[
    (df_test['upset_signal_score'] >= 0.4) &  # Segnali forti di upset
    (df_test['upset_signal'] == True) &  # Modello predice upset
    (df_test['match_reliability'] >= 0.5) &  # Partita mediamente prevedibile
    (df_test['max_odds'] >= 2.0)  # Value odds
]
roi, acc, n = calculate_strategy_roi(subset)
advanced_strategies.append({
    'Strategy': 'ValueUpset: HighSignal+Reliable+Value',
    'Bets': n,
    'Accuracy': acc,
    'ROI': roi,
    'Type': 'Value'
})

# Strategia E: Super Safe - tutto allineato
subset = df_test[
    (df_test['match_reliability'] >= 0.6) &
    (df_test['confidence'] >= 0.7) &
    (df_test['upset_signal_score'] <= 0.15) &
    (df_test['min_odds'] <= 1.5)  # Favorito forte
]
roi, acc, n = calculate_strategy_roi(subset, bet_on_model=False)  # Scommetti sempre sul favorito
advanced_strategies.append({
    'Strategy': 'SuperSafe: AllAligned+StrongFav',
    'Bets': n,
    'Accuracy': acc,
    'ROI': roi,
    'Type': 'UltraSafe'
})

# Strategia F: Giocatori TOP affidabili (>65% accuracy)
top_reliable_players = [p for p, r in player_reliability.items() if r >= 0.65]
subset = df_test[
    (df_test['Player_1'].isin(top_reliable_players) | df_test['Player_2'].isin(top_reliable_players)) &
    (df_test['confidence'] >= 0.5)
]
roi, acc, n = calculate_strategy_roi(subset)
advanced_strategies.append({
    'Strategy': 'TopReliablePlayers (>65% acc) + Conf>=50%',
    'Bets': n,
    'Accuracy': acc,
    'ROI': roi,
    'Type': 'PlayerBased'
})

# Strategia G: Evita giocatori imprevedibili (<45% accuracy)
unreliable_players = [p for p, r in player_reliability.items() if r < 0.45]
subset = df_test[
    ~(df_test['Player_1'].isin(unreliable_players) | df_test['Player_2'].isin(unreliable_players)) &
    (df_test['confidence'] >= 0.6)
]
roi, acc, n = calculate_strategy_roi(subset)
advanced_strategies.append({
    'Strategy': 'AvoidUnreliable (<45% acc) + Conf>=60%',
    'Bets': n,
    'Accuracy': acc,
    'ROI': roi,
    'Type': 'PlayerBased'
})

# Strategia H: Combinata ottimale avanzata
subset = df_test[
    (df_test['composite_score'] >= 0.55) &
    (df_test['match_reliability'] >= 0.55) &
    ((df_test['confidence'] >= 0.65) | ((df_test['upset_signal_score'] >= 0.35) & (df_test['max_odds'] >= 2.5)))
]
roi, acc, n = calculate_strategy_roi(subset)
advanced_strategies.append({
    'Strategy': 'OptimalCombo: Composite+Reliability+ConfOrValue',
    'Bets': n,
    'Accuracy': acc,
    'ROI': roi,
    'Type': 'Hybrid'
})

# Mostra risultati avanzati
adv_strat_df = pd.DataFrame(advanced_strategies).sort_values('ROI', ascending=False)
print("\n   STRATEGIE AVANZATE (Confidence + Upset Signal + Player Reliability):")
print(adv_strat_df.to_string(index=False))

# Combina tutte le strategie e salva
all_strategies = pd.concat([strat_df, adv_strat_df[['Strategy', 'Bets', 'Accuracy', 'ROI']]], ignore_index=True)
all_strategies = all_strategies.sort_values('ROI', ascending=False)
all_strategies.to_csv(f'{OUTPUT_DIR}/betting_strategies.csv', index=False)

# Analisi correlazioni indicatori
print("\n   Correlazione tra indicatori:")
corr_cols = ['confidence', 'match_reliability', 'upset_signal_score', 'composite_score', 'correct']
print(df_test[corr_cols].corr().round(3).to_string())

# Top 10 partite per composite score
print("\n   Top 10 partite per Composite Score:")
top_composite = df_test.nlargest(10, 'composite_score')[
    ['Player_1', 'Player_2', 'confidence', 'match_reliability', 'upset_signal_score', 'composite_score', 'correct']
]
print(top_composite.to_string(index=False))

# ==============================================================================
# 9C) ANALISI ANNO PER ANNO (2021-2025)
# ==============================================================================
print("\n" + "="*80)
print("9C) VALIDAZIONE STRATEGIE ANNO PER ANNO (2021-2025)")
print("="*80)

# Estrai anno dalla data
df_test['Year'] = pd.to_datetime(df_test['Date']).dt.year

# Definisci le migliori strategie da testare anno per anno
best_strategies = {
    'Reliability>=65% + Conf>=50%': lambda df: df[(df['match_reliability'] >= 0.65) & (df['confidence'] >= 0.5)],
    'Reliability>=60% + Conf>=50%': lambda df: df[(df['match_reliability'] >= 0.60) & (df['confidence'] >= 0.5)],
    'Reliability>=55% + Conf>=60%': lambda df: df[(df['match_reliability'] >= 0.55) & (df['confidence'] >= 0.6)],
    'TopReliablePlayers + Conf>=50%': lambda df: df[(df['Player_1'].isin(top_reliable_players) | df['Player_2'].isin(top_reliable_players)) & (df['confidence'] >= 0.5)],
    'Solo favoriti odds<=1.3': lambda df: df[df['min_odds'] <= 1.3],
    'Baseline (tutte)': lambda df: df,
}

years_to_analyze = [2021, 2022, 2023, 2024, 2025]
yearly_results = []

for year in years_to_analyze:
    df_year = df_test[df_test['Year'] == year]
    if len(df_year) < 50:
        continue
    
    print(f"\n   === ANNO {year} ({len(df_year)} partite nel test set) ===")
    
    for strat_name, strat_func in best_strategies.items():
        subset = strat_func(df_year)
        if len(subset) >= 10:
            roi, acc, n = calculate_strategy_roi(subset)
            yearly_results.append({
                'Year': year,
                'Strategy': strat_name,
                'Bets': n,
                'Accuracy': acc,
                'ROI': roi
            })
            print(f"   {strat_name}: {n} bets, Acc={acc:.1%}, ROI={roi:+.1f}%")

# Crea tabella pivot per confronto
if yearly_results:
    yearly_df = pd.DataFrame(yearly_results)
    
    print("\n" + "="*80)
    print("   RIEPILOGO ROI PER ANNO E STRATEGIA:")
    print("="*80)
    
    pivot_roi = yearly_df.pivot_table(index='Strategy', columns='Year', values='ROI', aggfunc='first')
    pivot_roi['Media'] = pivot_roi.mean(axis=1)
    pivot_roi['Consistenza'] = pivot_roi.iloc[:, :-1].apply(lambda x: (x > 0).sum(), axis=1)
    pivot_roi = pivot_roi.sort_values('Media', ascending=False)
    print(pivot_roi.round(1).to_string())
    
    print("\n   RIEPILOGO ACCURACY PER ANNO E STRATEGIA:")
    pivot_acc = yearly_df.pivot_table(index='Strategy', columns='Year', values='Accuracy', aggfunc='first')
    pivot_acc['Media'] = pivot_acc.mean(axis=1)
    pivot_acc = pivot_acc.sort_values('Media', ascending=False)
    print((pivot_acc * 100).round(1).to_string())
    
    # Salva analisi annuale
    yearly_df.to_csv(f'{OUTPUT_DIR}/yearly_strategy_analysis.csv', index=False)
    pivot_roi.to_csv(f'{OUTPUT_DIR}/yearly_roi_pivot.csv')
    
    print(f"\n   Salvati: yearly_strategy_analysis.csv, yearly_roi_pivot.csv")

# ==============================================================================
# 12) ANALISI QUOTE E CONFIDENCE
# ==============================================================================
print("\n" + "="*80)
print("10) ANALISI DETTAGLIATA QUOTE vs CONFIDENCE")
print("="*80)

# Bins di confidence
conf_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
df_test['conf_bin'] = pd.cut(df_test['confidence'], bins=conf_bins)

# Bins di quote (min odds = favorito)
odds_bins = [1.0, 1.3, 1.5, 2.0, 3.0, 10.0]
df_test['odds_bin'] = pd.cut(df_test['min_odds'], bins=odds_bins)

# Cross-tab analysis
print("\n   Accuracy per Confidence Bin:")
conf_analysis = df_test.groupby('conf_bin').agg({
    'correct': ['mean', 'sum', 'count'],
    'upset_signal': 'mean'
}).round(3)
conf_analysis.columns = ['Accuracy', 'Correct', 'Total', 'Upset_Rate']
print(conf_analysis.to_string())

print("\n   Accuracy per Odds Bin (Favoriti):")
odds_analysis = df_test.groupby('odds_bin').agg({
    'correct': ['mean', 'sum', 'count'],
    'upset_signal': 'mean'
}).round(3)
odds_analysis.columns = ['Accuracy', 'Correct', 'Total', 'Upset_Rate']
print(odds_analysis.to_string())

# ==============================================================================
# 13) SALVATAGGIO E VISUALIZZAZIONI
# ==============================================================================
print("\n" + "="*80)
print("11) SALVATAGGIO RISULTATI E VISUALIZZAZIONI")
print("="*80)

# Salva modelli
joblib.dump(scaler, f'{OUTPUT_DIR}/scaler.joblib')
joblib.dump(encoder, f'{OUTPUT_DIR}/encoder.joblib')

# Plot 1: Training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(train_losses, label='Train Loss')
axes[0, 0].plot(val_losses, label='Val Loss')
axes[0, 0].set_title('Loss over epochs')
axes[0, 0].legend()

axes[0, 1].plot(train_aucs, label='Train AUC')
axes[0, 1].plot(val_aucs, label='Val AUC')
axes[0, 1].set_title('AUC over epochs')
axes[0, 1].legend()

# Plot 2: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# Plot 3: Calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(test_labels, test_preds, n_bins=10)
axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
axes[1, 1].plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
axes[1, 1].set_title('Calibration Curve')
axes[1, 1].set_xlabel('Mean Predicted Value')
axes[1, 1].set_ylabel('Fraction of Positives')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_analysis.png', dpi=150)
plt.close()

# Plot 4: Distribuzione accuratezza per giocatore
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram accuratezza
axes[0, 0].hist(player_df['Accuracy'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(player_df['Accuracy'].mean(), color='red', linestyle='--', label=f'Mean: {player_df["Accuracy"].mean():.2%}')
axes[0, 0].set_title('Distribuzione Accuratezza per Giocatore')
axes[0, 0].set_xlabel('Accuracy')
axes[0, 0].set_ylabel('Numero Giocatori')
axes[0, 0].legend()

# Scatter: Upset Caused vs Suffered
axes[0, 1].scatter(player_df['Upset_Caused_Rate'], player_df['Upset_Suffered_Rate'], 
                   alpha=0.5, s=player_df['Matches']*2)
axes[0, 1].set_title('Upset Caused vs Suffered')
axes[0, 1].set_xlabel('Upset Caused Rate')
axes[0, 1].set_ylabel('Upset Suffered Rate')

# Bar plot superfici
surface_plot = surface_df.set_index('Surface')['Accuracy'].plot(kind='bar', ax=axes[1, 0], color='steelblue')
axes[1, 0].set_title('Accuracy per Superficie')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].axhline(test_acc, color='red', linestyle='--', label=f'Overall: {test_acc:.2%}')
axes[1, 0].legend()
plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

# ROI per strategia
strat_plot = strat_df.set_index('Strategy')['ROI'].head(10).plot(kind='barh', ax=axes[1, 1], color='green')
axes[1, 1].set_title('ROI per Strategia (Top 10)')
axes[1, 1].set_xlabel('ROI (%)')
axes[1, 1].axvline(0, color='red', linestyle='--')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/detailed_analysis.png', dpi=150)
plt.close()

# ==============================================================================
# SUMMARY FINALE
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY FINALE")
print("="*80)

print(f"""
METRICHE GLOBALI:
  - Test Accuracy: {test_acc:.2%}
  - Test AUC: {test_auc:.4f}
  - Test Precision: {precision_score(test_labels_int, test_preds_binary):.2%}
  - Test Recall: {recall_score(test_labels_int, test_preds_binary):.2%}
  - Brier Score: {brier_score_loss(test_labels_int, test_preds):.4f}

ANALISI SUPERFICIE:
  - Migliore: {surface_df.iloc[0]['Surface']} ({surface_df.iloc[0]['Accuracy']:.2%})
  - Peggiore: {surface_df.iloc[-1]['Surface']} ({surface_df.iloc[-1]['Accuracy']:.2%})

ANALISI GIOCATORI:
  - Giocatori analizzati: {len(player_df)}
  - Accuracy media per giocatore: {player_df['Accuracy'].mean():.2%}
  - Giocatore più prevedibile: {easy_players.iloc[0]['Player']} ({easy_players.iloc[0]['Accuracy']:.2%})
  - Giocatore meno prevedibile: {hard_players.iloc[0]['Player']} ({hard_players.iloc[0]['Accuracy']:.2%})

MIGLIOR STRATEGIA BETTING:
  - {strat_df.iloc[0]['Strategy']}: ROI={strat_df.iloc[0]['ROI']:.2f}%, Accuracy={strat_df.iloc[0]['Accuracy']:.2%}

FILE SALVATI:
  - {OUTPUT_DIR}/best_model.pt
  - {OUTPUT_DIR}/surface_analysis.csv
  - {OUTPUT_DIR}/player_analysis.csv
  - {OUTPUT_DIR}/betting_strategies.csv
  - {OUTPUT_DIR}/training_analysis.png
  - {OUTPUT_DIR}/detailed_analysis.png
""")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
