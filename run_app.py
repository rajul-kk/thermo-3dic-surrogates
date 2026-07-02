import os
import uvicorn

if __name__ == '__main__':
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '8000'))
    reload = os.environ.get('RELOAD', 'true').lower() == 'true'
    uvicorn.run('app.main:app', host=host, port=port, reload=reload)
