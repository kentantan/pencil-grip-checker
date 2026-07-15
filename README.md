# えんぴつの持ち方見守り

スマホのブラウザ内でMediaPipe Hand Landmarkerを実行し、正しい持ち方との差を判定する試作Webアプリです。

## 個人データ

- カメラ映像・写真・動画は保存しません。
- 外部サーバーへ画像や手の座標を送信するコードはありません。
- 登録するのは正規化した手の特徴量だけで、IndexedDBへ端末内保存します。
- MediaPipeライブラリ・WASM・AIモデルも同一GitHub Pagesサイトから読み込みます。
- GitHub Pagesへの通常のアクセス通信は発生し、GitHub側ではIPアドレスがセキュリティ目的で記録されます。

## 準備（Windows 11）

1. PowerShellを開き、このフォルダへ移動します。
2. 次を実行します。

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_vendor.ps1
```

3. `vendor/mediapipe` と `models/hand_landmarker.task` が作成されたことを確認します。

## GitHub Pagesへ公開

1. GitHubで公開リポジトリを作成します。
2. このフォルダ内の全ファイルをリポジトリ直下へアップロードします。
3. `Settings` → `Pages` を開きます。
4. `Deploy from a branch`、`main`、`/(root)` を選んで保存します。
5. 表示されたHTTPSのURLをスマホで開きます。

## 使い方

1. 「カメラを開始」を押し、カメラを許可します。
2. スマホを斜め上から手元が映る位置に固定します。
3. 「10秒登録を始める」を押し、正しく持って文字を書きます。
4. 登録後に「見守りを開始」を押します。
5. 誤判定が多いときは「判定の厳しさ」を調整します。

## 注意

この試作は鉛筆自体を検出せず、手の21点の形から登録時との差を判定します。親指が鉛筆にどう接触しているかなど、見えない情報は判定できません。

## 外部送信コードの簡易確認

```powershell
.\verify_privacy.ps1
```

これはアプリ本体に外部URLや `fetch`、`WebSocket`、`sendBeacon` などが入っていないかを確認します。完全なセキュリティ監査を保証するものではありません。
