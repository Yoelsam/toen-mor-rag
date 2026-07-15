const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode');
const express = require('express');
const http = require('http');

const app = express();
let qrCodeData = '';
let isReady = false;
let clientInfo = null;

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: '/home/rag/whatsapp-bot/session' }),
  puppeteer: { 
    headless: true, 
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
  }
});

client.on('qr', async (qr) => {
  console.log('QR code recu - scannez depuis /whatsapp-qr');
  qrCodeData = await qrcode.toDataURL(qr);
});

client.on('ready', () => {
  isReady = true;
  clientInfo = client.info;
  console.log('WhatsApp connecte: ' + client.info.pushname);
});

client.on('message', async (msg) => {
  try {
    const contact = await msg.getContact();
    const chat = await msg.getChat();
    const phone = msg.from.replace('@c.us', '').replace('@g.us', '');
    const name = contact.pushname || contact.name || phone;
    const isGroup = msg.from.endsWith('@g.us');
    const groupName = isGroup ? chat.name : '';

    console.log('Message de ' + name + ' (' + phone + '): ' + (msg.body || msg.type));

    const payload = {
      entry: [{
        changes: [{
          value: {
            messages: [{
              from: phone,
              type: msg.type === 'chat' ? 'text' : msg.type,
              text: msg.type === 'chat' ? { body: msg.body } : undefined
            }],
            contacts: [{ profile: { name: isGroup ? groupName : name } }]
          }
        }]
      }]
    };

    if (msg.hasMedia) {
      const media = await msg.downloadMedia();
      if (media) {
        payload.media = {
          data: media.data,
          mimetype: media.mimetype,
          filename: media.filename || (msg.type + '.' + media.mimetype.split('/')[1])
        };
      }
    }

    const postData = JSON.stringify(payload);
    const req = http.request({
      hostname: '127.0.0.1',
      port: 8001,
      path: '/whatsapp-drive',
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(postData) }
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => console.log('Drive: ' + data));
    });
    req.on('error', (e) => console.error('Erreur Drive: ' + e.message));
    req.write(postData);
    req.end();

  } catch (err) {
    console.error('Erreur message: ' + err.message);
  }
});

app.get('/whatsapp-qr', (req, res) => {
  if (isReady) {
    res.send('<html dir="rtl"><body style="font-family:Arial;text-align:center;padding:40px;"><h1 style="color:green;">WhatsApp connecte!</h1><p>Connecte en tant que: ' + (clientInfo ? clientInfo.pushname : '') + '</p></body></html>');
  } else if (qrCodeData) {
    res.send('<html dir="rtl"><body style="font-family:Arial;text-align:center;padding:40px;"><h1>Scannez le QR Code</h1><p>Ouvrez WhatsApp > Appareils lies > Lier un appareil</p><img src="' + qrCodeData + '" style="width:300px;"/><script>setTimeout(function(){location.reload()},20000)</script></body></html>');
  } else {
    res.send('<html dir="rtl"><body style="font-family:Arial;text-align:center;padding:40px;"><h1>En attente du QR code...</h1><script>setTimeout(function(){location.reload()},5000)</script></body></html>');
  }
});

app.get('/whatsapp-status', (req, res) => {
  res.json({ connected: isReady, user: clientInfo ? clientInfo.pushname : null });
});

app.listen(3000, () => console.log('Serveur QR sur port 3000'));
client.initialize();
