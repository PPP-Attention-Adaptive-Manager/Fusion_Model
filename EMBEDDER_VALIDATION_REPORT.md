# Embedder Validation Report (pre-Tucker gate)

**But :** valider chaque embedder **isolément** sur données brutes réelles AVANT toute reconstruction du
Tucker. Le Tucker est *interactionnel* (produit externe des 4 modalités) : un seul embedder invalide ou
dégénéré **contamine tous les slices**. Donc on « goûte chaque ingrédient » avant de cuisiner.

**Script :** [scripts/embedders/validate_embedders.py](scripts/embedders/validate_embedders.py)
**Sortie :** [outputs/embedders/validation/embedder_validation.json](outputs/embedders/validation/embedder_validation.json)
(+ `keyboard_embeddings.npy`, `mouse_embeddings.npy`, `*_windows.csv`)

## Verdict global

| | verdict | entraîné ? | utilisable dans Tucker ? |
|---|---|---|---|
| **keyboard** | ⚠️ **WARNING** | oui | non (qualité insuffisante) |
| **mouse** | 🔴 **FAIL** | **non** | non (non entraîné) |

> `tucker_rebuild_allowed = false` → **🔴 RECONSTRUCTION DU TUCKER BLOQUÉE.**

---

## Keyboard — WARNING (entraîné mais embedding faible)

Encodeur `KeystrokeEncoder` LSTM, **3303 fenêtres** réelles (63 sessions), 20 keystrokes/fenêtre.

| Métrique | Valeur | Lecture |
|---|---|---|
| NaN / Inf | aucun | ✅ sain |
| cold-start (zéros) | 0 / 3303 | ✅ produit toujours une sortie |
| duplicate_ratio | 0.00 | ✅ pas d'effondrement vers un vecteur unique |
| variance moyenne | **5.6e-05** | ⚠️ très faible |
| norme L2 moyenne | **0.056** | ⚠️ embeddings minuscules |
| stabilité temporelle (cos consécutif) | **0.034** | ⚠️ fenêtres voisines quasi orthogonales (aucune continuité) |
| **corrélation comportementale (best \|r\|)** | **0.07** (typing_speed) | 🔴 **quasi nulle** |

**Diagnostic** : techniquement vivant (pas de NaN, pas de collapse en doublons), **mais l'embedding ne
porte presque aucun signal comportemental** : il ne corrèle pas avec la vitesse de frappe (|r|=0.07), a une
norme et une variance minuscules, et n'a aucune continuité temporelle. **Cause racine** : l'objectif
d'entraînement est **trivial** — la loss contrastive traite *chaque fenêtre comme sa propre classe*
(`labels = arange(batch)`), ce qui donne 100 % d'« accuracy » factice et n'apprend aucune structure utile.
→ **À ré-entraîner avec un vrai objectif** (vraies paires positives, ex. augmentations de fenêtres /
fenêtres voisines du même utilisateur).

## Mouse — FAIL (non entraîné)

Encodeur `MouseEncoderP2`, échantillon mécanique borné de **40 fenêtres** (4 sessions), CPU.

| Métrique | Valeur | Lecture |
|---|---|---|
| `trained / pretrained` | **false** | 🔴 **gate : un encodeur non entraîné ne peut pas PASS** |
| cold-start (zéros) | **32 / 40** | 🔴 fragile : 80 % des fenêtres ne produisent rien d'exploitable |
| fenêtres valides | **8** (< 30) | ⚠️ trop peu pour juger |
| NaN / Inf | aucun | ✅ |
| duplicate_ratio | 0.25 | ⚠️ |
| stabilité temporelle (cos consécutif) | **0.946** | ⚠️ sortie quasi constante |
| corrélation comportementale (best \|r\|) | 0.86 (n_events) | ⚠️ **trompeur — voir note** |

**Note importante sur le 0.86** : cette corrélation « forte » **n'est pas une preuve de qualité**. Le TCN
de `MouseEncoderP2` utilise des **filtres déterministes figés** (Gabor / gradient) qui laissent passer le
signal cinématique brut même sans entraînement — donc la sortie corrèle mécaniquement avec `n_events` /
vitesse. Mais les **parties entraînables** (projection TCN, click-enc, stats-MLP, fusion) sont **non
entraînées**, la sortie est quasi constante dans le temps (cos 0.946) et 80 % des fenêtres sont vides.
**Ce n'est pas une représentation apprise.** Verdict : **FAIL** (confirmé par le gate `trained=false`).

> Robustesse à corriger plus tard : 32/40 fenêtres en cold-start indique aussi que le câblage
> extraction→encodeur mouse devra être fiabilisé avant usage réel.

---

## Décision

1. **Ne pas reconstruire le Tucker** (plan A « mouse seedé » définitivement rejeté — confirmé par les
   chiffres : ni mouse ni keyboard ne sont des experts valides).
2. **Mouse** = blocage principal → écrire un **vrai objectif d'entraînement** (aucun n'existe), entraîner,
   puis re-valider jusqu'à PASS.
3. **Keyboard** = ré-entraîner avec un objectif contrastif **non trivial** (vraies paires positives), puis
   re-valider jusqu'à PASS.
4. **Notif** + **switching** : déjà opérationnels (réels), à valider de la même manière par sécurité.
5. Quand **mouse ET keyboard = PASS** → débloquer la reconstruction du Tucker complet (étapes 3→7 du plan).

**Critères de PASS (re-validation)** : `trained=true`, pas de NaN/Inf, cold-start ≈ 0, `duplicate_ratio`
bas, variance non dégénérée, stabilité temporelle réaliste (ni ≈0 ni ≈1), et **corrélation comportementale
\|r\| ≥ 0.2** avec au moins une feature simple.

## Reproduire

```powershell
# validation complète
python scripts/embedders/validate_embedders.py --raw-data-dir data --device cuda
# re-valider uniquement le mouse (réutilise le clavier sauvegardé)
python scripts/embedders/validate_embedders.py --raw-data-dir data --skip-keyboard
```
