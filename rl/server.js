const http = require('http');
const FalconEnv = require('./falcon_env');

const env = new FalconEnv();

const server = http.createServer((req, res) => {
    if (req.method === 'POST') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', () => {
            try {
                const data = JSON.parse(body);

                if (req.url === '/reset') {
                    const obs = env.reset();
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ state: obs }));
                }
                else if (req.url === '/step') {
                    const action = data.action; // Expecting [throttle, gimbal, rcs]
                    const result = env.step(action);
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify(result));
                }
                else {
                    res.writeHead(404);
                    res.end('Not Found');
                }
            } catch (e) {
                res.writeHead(500);
                res.end('Error: ' + e.message);
            }
        });
    } else {
        res.writeHead(405);
        res.end('Method Not Allowed');
    }
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Falcon RL Server running on port ${PORT}`);
});
