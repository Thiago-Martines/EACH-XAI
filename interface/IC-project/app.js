var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
var bodyParser = require('body-parser');
var { spawn } = require('child_process');
var multer = require('multer');
var fs = require('fs');
var WebSocket = require('ws');
var os = require('os');
var http = require('http');

// Define route var
var usersRouter = require('./routes/users');
var indexRouter = require('./routes/index');
var modelDevRouter = require('./routes/modelDev');
var modelAppRouter = require('./routes/modelApp');

var app = express();
const wss = new WebSocket.Server({ port: 3001 });

let currentWebSocket = null;

wss.on('connection', (ws) => {
    console.log('Cliente conectado via WebSocket');
    currentWebSocket = ws;
    
    ws.on('close', () => {
        console.log('Cliente desconectado via WebSocket');
        currentWebSocket = null;
    });
});

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

app.use(bodyParser.json());
app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

//-------------------------------------- File Handler -----------------------\\
// Set storage for multer
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    // Determine destination folder based on route
    if (req.path === '/uploadDataset') {
      clearFolder('./public/dataset', () => {
        cb(null, './public/dataset');
      });
    } else if (req.path === '/uploadModel') {
      clearFolder('./public/model', () => {
        cb(null, './public/model');
      });
    } else if (req.path === '/uploadSample') {
      clearFolder('./public/sample', () => {
        cb(null, './public/sample');
      });
    } else {
      cb(new Error('Invalid upload route'));
    }
  },
  filename: function (req, file, cb) {
    cb(null, file.originalname);
  }
});

// Multer upload
const upload = multer({ storage: storage });

// Function to clear folder
function clearFolder(folderPath, callback) {
  fs.readdir(folderPath, (err, files) => {
    if (err) throw err;

    for (const file of files) {
      fs.unlink(path.join(folderPath, file), err => {
        if (err) throw err;
      });
    }

    callback();
  });
}

// Define route for handling dataset file upload
app.post('/uploadDataset', upload.single('uploadDataset'), (req, res) => {
  // Check for upload errors
  if (req.fileValidationError) {
    return res.status(400).send('Invalid file format');
  } else if (!req.file) {
    return res.status(400).send('No file uploaded');
  }

  // Move the uploaded file to the desired location
  const filePath = path.join(__dirname, 'public/dataset', req.file.originalname);
  fs.renameSync(req.file.path, filePath);

  // Send an empty response with success status code
  res.status(200).send();
});

// Define route for handling other type of file upload
app.post('/uploadModel', upload.single('uploadModel'), (req, res) => {
  // Check for upload errors
  if (req.fileValidationError) {
    return res.status(400).send('Invalid file format');
  } else if (!req.file) {
    return res.status(400).send('No file uploaded');
  }

  // Move the uploaded file to the desired location
  const filePath = path.join(__dirname, 'public/model', req.file.originalname);
  fs.renameSync(req.file.path, filePath);

  // Send an empty response with success status code
  res.status(200).send();
});
//----------------------------------------------------------------------------------\\

// Define route
app.use('/', indexRouter);
app.use('/users', usersRouter);
app.use('/modelDev', modelDevRouter);
app.use('/modelApp', modelAppRouter);

//----------------------------------- Get File Name --------------------------------\\
app.get('/datasetFileName', (req, res) => {
  const datasetPath = path.join(__dirname, 'public', 'dataset');

  fs.readdir(datasetPath, (err, files) => {
    if (err) {
      return res.status(500).send('Erro ao ler o diretório do dataset');
    }

    const datasetFiles = files.filter(file => file.endsWith('.csv') || file.endsWith('.xlsx'));

    if (datasetFiles.length === 0) {
      return res.status(404).send('Nenhum dataset selecionado');
    }
    
    const fileName = datasetFiles[0]; //Assumindo apenas um arquivo na pasta

    // Executa o script Python
    const pythonProcess = spawn('python', ['dataset.py', fileName]);

    let output = ''; // Para acumular a saída do script Python

    // Captura a saída do script Python
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    // Captura erros do script Python
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Erro do script: ${data}`);
    });

    // Lida com o término do processo Python
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        return res.status(500).send('Erro ao executar o script Python');
      }

      try {
        // Analisa a saída como JSON
        const parsedOutput = JSON.parse(output);

        // Retorna o fileName, numVariaveis e a lista de variáveis
        res.json({
          fileName,
          numVariaveis: parsedOutput.num_variaveis,
          variaveis: parsedOutput.variaveis,
          descritiva_base: parsedOutput.descritiva_base
        });
      } catch (error) {
        console.error('Erro ao analisar o JSON:', error);
        return res.status(500).send('Erro ao processar a saída do script Python');
      }
    });
  });
});

app.get('/modelFileName', (req, res) => {
  const modelPath = path.join(__dirname, 'public', 'model');

  fs.readdir(modelPath, (err, files) => {
    if (err) {
      return res.status(500).send('Erro ao ler o diretório do modelo');
    }

    if (files.length === 0) {
      return res.status(404).send('nenhum modelo selecionado');
    }

    const fileName = files[0]; // Assumindo apenas um arquivo na pasta

    res.json({ fileName});
  });
});
//----------------------------------------------------------------------------------\\


//------------------------------------ Python Scripts -------------------------------\\

// Endpoint para processar os dados selecionados e chamar o script Python
app.post("/processSelectedData", (req, res) => {
  const selectedData = req.body['selectedData[]'];
  // Chamar o script Python com os dados selecionados como argumentos
  const pythonProcess = spawn('python', ['manual_filter.py', ...selectedData]);

  let csvPath = '';

  pythonProcess.stdout.on('data', (data) => {
    // Concatena os dados de saída do script Python
    csvPath += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
      const message = data.toString();
      console.error('Erro no script Python:', message);

      // Verifica se é uma mensagem de progresso
      if (message.includes('[PROGRESS]')) {
          if (currentWebSocket && currentWebSocket.readyState === WebSocket.OPEN) {
              console.log('Enviando mensagem via WebSocket:', message); // Para debug
              currentWebSocket.send(JSON.stringify({ 
                  type: 'progress', 
                  message: message.replace('[PROGRESS] ', '') 
              }));
          } else {
              console.log('WebSocket não está conectado'); // Para debug
          }
      }
  });

  pythonProcess.stdout.on('data', (data) => {
    // Captura a saída do script Python (o dataset em formato CSV)
    const datasetCsv = data.toString();

    // Envia o dataset como resposta para o cliente
    res.send(datasetCsv);
  });

  pythonProcess.stderr.on('data', (data) => {
    // Em caso de erro no script Python, você pode lidar com ele aqui
    console.error("Erro ao executar o script Python:", data);
  });
});

app.post('/trainGridModel', (req, res) => { 
  const conteudoFinal = req.body.conteudoFinal;

  const fs = require('fs');
  const filePath = './temp_config.txt';

  fs.writeFile(filePath, conteudoFinal, (err) => {
    if (err) {
      return res.status(500).send('Erro ao salvar o conteúdo no arquivo');
    }

    const pythonProcess = spawn('python', ['grid.py', filePath]);

    let output = '';

    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    let stderrBuffer = '';

    pythonProcess.stderr.on('data', (data) => {
        stderrBuffer += data.toString();
        
        // Processa linhas completas
        let lines = stderrBuffer.split('\n');
        stderrBuffer = lines.pop();
        
        lines.forEach(line => {
            if (line.trim() === '') return;
            
            console.error('Erro no script Python:', line);
            
            if (line.includes('[PROGRESS]')) {
                if (currentWebSocket && currentWebSocket.readyState === WebSocket.OPEN) {
                    currentWebSocket.send(JSON.stringify({ 
                        type: 'progress', 
                        message: line.replace('[PROGRESS] ', '') 
                    }));
                }
            }
            
            if (line.includes('[TRAINING]')) {
                if (currentWebSocket && currentWebSocket.readyState === WebSocket.OPEN) {
                    currentWebSocket.send(JSON.stringify({ 
                        type: 'training', 
                        message: line.replace('[TRAINING] ', '') 
                    }));
                }
            }
        });
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
          try {
              // Log do output do script Python
              console.log("Raw output do Python:", output);
  
              // Parse o JSON retornado pelo Python
              const parsedOutput = JSON.parse(output);
  
              // Log do resultado parseado
              console.log("Output parseado:", parsedOutput);
  
              res.json(parsedOutput);
          } catch (parseError) {
              console.error("Erro ao interpretar JSON:", parseError);
              res.status(500).send('Erro ao interpretar o resultado do script Python');
          }
      } else {
          console.error("Erro no script Python, código:", code);
          res.status(500).send('Erro ao executar o script Python');
      }
  
      // fs.unlinkSync(filePath);
  });
  });
});

// Explicação SHAP
app.post('/shapEXP', (req, res) => {
  console.log('Iniciando processo de explicação SHAP...');
  
  // Executa o script Python para gerar explicações SHAP
  const pythonProcess = spawn('python', ['shapEXP.py']);

  let output = '';

  pythonProcess.stdout.on('data', (data) => {
    output += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    const message = data.toString();
    console.error('Erro no script SHAP:', message);

    // Verifica se é uma mensagem de progresso
    if (message.includes('[PROGRESS]')) {
      if (currentWebSocket && currentWebSocket.readyState === WebSocket.OPEN) {
        console.log('Enviando progresso SHAP via WebSocket:', message);
        currentWebSocket.send(JSON.stringify({ 
          type: 'shap_progress', 
          message: message.replace('[PROGRESS] ', '') 
        }));
      }
    }
  });

  pythonProcess.on('close', (code) => {
    if (code === 0) {
      try {
        console.log("Raw output do script SHAP:", output);
        
        // Parse o JSON retornado pelo script Python
        const parsedOutput = JSON.parse(output);
        
        console.log("Output SHAP parseado:", parsedOutput);
        
        // Retorna o JSON completo para o frontend
        res.json(parsedOutput);
      } catch (parseError) {
        console.error("Erro ao interpretar JSON do SHAP:", parseError);
        res.status(500).send('Erro ao interpretar o resultado do script SHAP');
      }
    } else {
      console.error("Erro no script SHAP, código:", code);
      res.status(500).send('Erro ao executar o script SHAP');
    }
  });
});
// Explicação LIME
app.post('/limeEXP', (req, res) => {
  console.log('Iniciando processo de explicação LIME...');
  
  // Executa o script Python para gerar explicações LIME
  const pythonProcess = spawn('python', ['limeEXP.py']);

  let output = '';

  pythonProcess.stdout.on('data', (data) => {
    output += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    const message = data.toString();
    console.error('Erro no script LIME:', message);

    // Verifica se é uma mensagem de progresso
    if (message.includes('[PROGRESS]')) {
      if (currentWebSocket && currentWebSocket.readyState === WebSocket.OPEN) {
        console.log('Enviando progresso LIME via WebSocket:', message);
        currentWebSocket.send(JSON.stringify({ 
          type: 'lime_progress', 
          message: message.replace('[PROGRESS] ', '') 
        }));
      }
    }
  });

  pythonProcess.on('close', (code) => {
    if (code === 0) {
      try {
        console.log("Raw output do script LIME:", output);
        
        // Parse o JSON retornado pelo script Python
        const parsedOutput = JSON.parse(output);
        
        console.log("Output LIME parseado:", parsedOutput);
        
        // Retorna o JSON completo para o frontend
        res.json(parsedOutput);
      } catch (parseError) {
        console.error("Erro ao interpretar JSON do LIME:", parseError);
        res.status(500).send('Erro ao interpretar o resultado do script LIME');
      }
    } else {
      console.error("Erro no script LIME, código:", code);
      res.status(500).send('Erro ao executar o script LIME');
    }
  });
});
//------------------------------------------------------------------------------\\

// catch 404 and forward to error handler
app.use(function (req, res, next) {
  next(createError(404));
});

// error handler
app.use(function (err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;