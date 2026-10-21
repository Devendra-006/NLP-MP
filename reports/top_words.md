# 🏷️ Highly Informative Features (Top Vocabulary per Category)

This table displays the **top 15 highest-weighted positive features** ($W_{k, j}$) learned by the Multinomial Logistic Regression model for each news category.

| Rank | 🌍 World | ⚽ Sports | 📈 Business | 🔬 Sci/Tech |
| :---: | :--- | :--- | :--- | :--- |
| **#1** | `afp` (+6.94) | `coach` (+6.73) | `oil` (+5.99) | `space` (+7.12) |
| **#2** | `iraq` (+6.77) | `cup` (+6.33) | `economy` (+4.89) | `internet` (+6.44) |
| **#3** | `president` (+5.41) | `team` (+6.30) | `tax` (+4.73) | `nasa` (+6.29) |
| **#4** | `afp afp` (+5.09) | `sports` (+6.02) | `bank` (+4.65) | `microsoft` (+5.87) |
| **#5** | `iraqi` (+5.07) | `league` (+5.58) | `economic` (+4.50) | `software` (+5.83) |
| **#6** | `nuclear` (+5.04) | `baseball` (+5.28) | `york reuters` (+4.13) | `scientists` (+5.64) |
| **#7** | `iran` (+4.88) | `players` (+5.08) | `company` (+4.11) | `reuters reuters` (+5.51) |
| **#8** | `un` (+4.68) | `season` (+4.96) | `inc` (+4.08) | `apple` (+5.14) |
| **#9** | `canadian press` (+4.62) | `39` (+4.92) | `airlines` (+4.03) | `online` (+5.06) |
| **#10** | `military` (+4.26) | `olympic` (+4.88) | `prices` (+4.03) | `web` (+4.82) |
| **#11** | `arafat` (+4.16) | `stadium` (+4.48) | `insurance` (+3.85) | `linux` (+4.76) |
| **#12** | `troops` (+4.15) | `football` (+4.12) | `corp` (+3.85) | `technology` (+4.72) |
| **#13** | `palestinian` (+4.11) | `nba` (+4.11) | `us` (+3.85) | `computer` (+4.59) |
| **#14** | `leader` (+4.09) | `athens` (+4.05) | `business` (+3.80) | `ibm` (+4.53) |
| **#15** | `israel` (+4.03) | `nascar` (+4.05) | `amp` (+3.73) | `study` (+4.14) |

---

### Linguistic & Domain Observations

1. **Sports**: Shows extraordinarily distinct lexical separation (`olympic`, `coach`, `cup`, `season`, `game`, `soccer`). This explains why Sports reaches the highest F1 score (**0.9660**).
2. **World**: Dominated by geopolitical and crisis vocabulary (`iraq`, `gaza`, `minister`, `troops`, `un`, `reuters`).
3. **Business**: Characterized by financial and corporate reporting terms (`oil`, `stocks`, `profit`, `shares`, `quarterly`, `economy`).
4. **Sci/Tech**: Driven by software, electronics, and digital technology terms (`software`, `microsoft`, `space`, `internet`, `apple`, `linux`).