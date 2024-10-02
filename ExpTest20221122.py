import sys
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from scipy import signal
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget, QAction, QFileDialog, \
    QVBoxLayout, QWidget, QPushButton, QGridLayout, QLabel, QInputDialog, \
    QLineEdit, QComboBox, QMessageBox, QCheckBox, QProgressBar, QHBoxLayout, QTableWidget, QTableWidgetItem, \
    QAbstractItemView, QHeaderView, QDialogButtonBox, QDialog, QGroupBox, QRadioButton, QButtonGroup
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

class MachineLearner:
    def __init__(self):
        self.modelLoaded = False

    def set(self, nnList, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb, callback):
        self.batchSize = batchSize
        self.epoch = epoch
        self.learningRate = learningRate
        self.splitPercentage = splitPercentage
        self.earlyStopping = earlyStopping
        self.verbose = verb
        self.callback = callback
        self.model = self.createModel(nnList)
        self.modelLoaded = True

    def fit(self, x_data, y_data):
        _callbacks = [self.callback]
        if self.earlyStopping == True:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=500)
            _callbacks.append(early_stopping)

        if self.splitPercentage > 0:
            training_history = self.model.fit(x_data, y_data, batch_size=self.batchSize, epochs=self.epoch,
                                              validation_split=self.splitPercentage, verbose=self.verbose,
                                              callbacks=[self.callback])
        else:
            training_history = self.model.fit(x_data, y_data, batch_size=self.batchSize, epochs=self.epoch,
                                              verbose=self.verbose, callbacks=_callbacks)

        return training_history

    def fitWithValidation(self, x_train_data, y_train_data, x_valid_data, y_valid_data):
        _callbacks = [self.callback]
        if self.earlyStopping == True:
            early_stopping = tf.keras.callbacks.EarlyStopping()
            _callbacks.append(early_stopping)

        training_history = self.model.fit(x_train_data, y_train_data, batch_size=self.batchSize, epochs=self.epoch,
                                          verbose=self.verbose, validation_data=(x_valid_data, y_valid_data),
                                          callbacks=_callbacks)

        return training_history

    def predict(self, x_data):
        y_predicted = self.model.predict(x_data)

        return y_predicted

    def createModel(self, nnList):
        adamOpt = tf.keras.optimizers.Adam(learning_rate=self.learningRate)
        model = tf.keras.Sequential()

        firstLayer = True
        for nn in nnList:
            noOfNeuron = nn[0]
            activationFunc = nn[1]
            if firstLayer:
                model.add(tf.keras.layers.Dense(units=noOfNeuron, activation=activationFunc, input_shape=[noOfNeuron]))
                firstLayer = False
            else:
                model.add(tf.keras.layers.Dense(units=noOfNeuron, activation=activationFunc))

        model.compile(loss='mse', optimizer=adamOpt)

        if self.verbose:
            model.summary()

        return model

    def saveModel(self, foldername):
        if self.modelLoaded == True:
            self.model.save(foldername)

    def loadModel(self, foldername):
        self.model = keras.models.load_model(foldername)
        self.modelLoaded = True

    def showResult(self, y_data, training_history, y_predicted, sensor_name, height):
        # max_val = max(y_predicted)
        index_at_max = max(range(len(y_predicted)), key=y_predicted.__getitem__)
        index_at_zero = index_at_max
        for i in range(index_at_max, len(y_predicted)-1):
            val = y_predicted[i]
            val_next = y_predicted[i+1]
            if val >= 0 and val_next <=0:
                index_at_zero = i
                break;

        fig, axs = plt.subplots(2, 1, figsize=(12, 12))
        title = sensor_name + height + ' / zeroIndexTime=' + str(index_at_zero * 0.000002)
        fig.suptitle(title)

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        axs[0].scatter(x_display, y_data, color="red", s=1)
        axs[0].plot(x_display, y_predicted, color='blue')
        axs[0].grid()

        lossarray = training_history.history['loss']
        axs[1].plot(lossarray, label='Loss')
        axs[1].grid()

        plt.show()

class LayerDlg(QDialog):
    def __init__(self, unit='128', af='relu'):
        super().__init__()
        self.initUI(unit, af)

    def initUI(self, unit, af):
        self.setWindowTitle('Machine Learning Curve Fitting/Interpolation')

        label1 = QLabel('Units', self)
        self.tbUnits = QLineEdit(unit, self)
        self.tbUnits.resize(100, 40)

        label2 = QLabel('Activation f', self)
        self.cbActivation = QComboBox(self)
        self.cbActivation.addItem(af)
        self.cbActivation.addItem('swish')
        self.cbActivation.addItem('relu')
        self.cbActivation.addItem('selu')
        self.cbActivation.addItem('sigmoid')
        self.cbActivation.addItem('softmax')

        if af == 'linear':
            self.cbActivation.setEnabled(False)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout = QGridLayout()
        layout.addWidget(label1, 0, 0)
        layout.addWidget(self.tbUnits, 0, 1)

        layout.addWidget(label2, 1, 0)
        layout.addWidget(self.cbActivation, 1, 1)

        layout.addWidget(self.buttonBox, 2, 1)

        self.setLayout(layout)

class DateDlg(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Select the date of Experiment')

        layout = QHBoxLayout()  # layout for the central widget
        widget = QWidget(self)  # central widget
        widget.setLayout(layout)

        number_group = QButtonGroup(widget)  # Number group
        self.radio20221115 = QRadioButton("2022-11-15")
        number_group.addButton( self.radio20221115)
        self.radio20221122 = QRadioButton("2022-11-22")
        number_group.addButton(self.radio20221122)
        self.radio202211Med = QRadioButton("2022-11-평균")
        number_group.addButton(self.radio202211Med)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.radio20221115)
        layout.addWidget(self.radio20221122)
        layout.addWidget(self.radio202211Med)
        layout.addWidget(buttonBox)

        self.resize(500, 40)

class MLWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # dateDlg = DateDlg()
        # rc = dateDlg.exec()
        #
        # if rc == 0:
        #     sys.exit(0)

        # if dateDlg.radio20221115.isChecked():
        #     self.distArrayName = ['S2m', 'S4m', 'S8m', 'S16m', 'S24m', 'S32m']
        #     self.distList = [2.0, 4.0, 8.0, 16.0, 24.0, 32.0]
        # elif dateDlg.radio20221122.isChecked() or dateDlg.radio202211Med.isChecked():
        self.distArrayName = ['S2m', 'S4m', 'S8m', 'S16m', 'S24m', 'S30m', 'S32m', 'S40m']
        self.distList = [2.0, 4.0, 8.0, 16.0, 24.0, 30.0, 32.0, 40.0]
        # else:
        #     QMessageBox.warning(self, 'Warning', 'No date is selected!')
        #     sys.exit(0)

        self.heightArrayName = ['H4.1m', 'H3.1m', 'H2.1m', 'H0.95m']
        self.heightList = [4.1, 3.1, 2.1, 0.95]

        self.initUI()

        self.indexijs = []
        self.time_data = None
        self.southSensors = []
        self.dataLoaded = False
        self.modelLearner = MachineLearner()

    def initMenu(self):
        # Menu
        openNN = QAction(QIcon('open.png'), 'Open NN', self)
        openNN.setStatusTip('Open Neural Network Structure from a File')
        openNN.triggered.connect(self.showNNFileDialog)

        saveNN = QAction(QIcon('save.png'), 'Save NN', self)
        saveNN.setStatusTip('Save Neural Network Structure in a File')
        saveNN.triggered.connect(self.saveNNFileDialog)

        exitMenu = QAction(QIcon('exit.png'), 'Exit', self)
        exitMenu.setStatusTip('Exit')
        exitMenu.triggered.connect(self.close)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openNN)
        fileMenu.addAction(saveNN)
        fileMenu.addSeparator()
        fileMenu.addAction(exitMenu)

        self.statusBar().showMessage('Welcome to Machine Learning')
        self.central_widget = QWidget()  # define central widget
        self.setCentralWidget(self.central_widget)  # set QMainWindow.centralWidget

    def initCSVFileReader(self):
        layout = QHBoxLayout()

        fileLabel = QLabel('csv file')
        self.editFile = QLineEdit('Please load DataRepository/KGS_2022_11_22_142943_0.2sec_filteredDataOnly - corrected.csv')
        self.editFile.setFixedWidth(700)
        openBtn = QPushButton('...')
        openBtn.clicked.connect(self.showFileDialog)

        layout.addWidget(fileLabel)
        layout.addWidget(self.editFile)
        layout.addWidget(openBtn)

        return layout

    def initSensor(self):
        layout = QGridLayout()

        rows = len(self.heightArrayName)
        cols = len(self.distArrayName)

        self.distArray = []
        for i in range(cols):
            cbDist = QCheckBox(self.distArrayName[i])
            cbDist.stateChanged.connect(self.distClicked)
            self.distArray.append(cbDist)

        self.heightArray = []
        for i in range(rows):
            cbHeight = QCheckBox(self.heightArrayName[i])
            cbHeight.stateChanged.connect(self.heightClicked)
            self.heightArray.append(cbHeight)

        self.cbArray = []
        for i in range(rows):
            col = []
            for j in range(cols):
                col.append(QCheckBox(''))
            self.cbArray.append(col)

        for i in range(len(self.distArray)):
            cbDist = self.distArray[i]
            layout.addWidget(cbDist, 0, i + 1)

        for i in range(len(self.heightArray)):
            cbHeight = self.heightArray[i]
            layout.addWidget(cbHeight, i + 1, 0)

        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray[i][j]
                qcheckbox.setTristate()
                layout.addWidget(qcheckbox, i + 1, j + 1)

        return layout

    def initReadOption(self):
        layout = QHBoxLayout()

        startRowLabel = QLabel('Start Row')
        startRowLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.startRow = QLineEdit('0')
        self.startRow.setFixedWidth(100)
        self.startRow.setEnabled(False)
        endRowLabel = QLabel('End Row')
        endRowLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.endRow = QLineEdit('100000')
        self.endRow.setFixedWidth(100)
        self.endRow.setEnabled(False)
        stepLabel = QLabel('step')
        stepLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.step = QLineEdit('1')
        self.step.setFixedWidth(100)
        self.step.setEnabled(False)
        loadButton = QPushButton('Load Data')
        loadButton.clicked.connect(self.loadData)
        showButton = QPushButton('Show Graph')
        showButton.clicked.connect(self.showGraphs)

        layout.addWidget(startRowLabel)
        layout.addWidget(self.startRow)
        layout.addWidget(endRowLabel)
        layout.addWidget(self.endRow)
        layout.addWidget(stepLabel)
        layout.addWidget(self.step)
        layout.addWidget(loadButton)
        layout.addWidget(showButton)

        return layout

    def initNNTable(self):
        layout = QGridLayout()

        # NN Table
        self.tableNNWidget = QTableWidget()
        self.tableNNWidget.setColumnCount(2)
        self.tableNNWidget.setHorizontalHeaderLabels(['Units', 'Activation'])

        # read default layers
        DEFAULT_LAYER_FILE = 'default.nn'
        self.updateNNList(DEFAULT_LAYER_FILE)

        self.tableNNWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableNNWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableNNWidget.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Button of NN
        btnAdd = QPushButton('Add')
        btnAdd.setToolTip('Add a Hidden Layer')
        btnAdd.clicked.connect(self.addLayer)
        btnEdit = QPushButton('Edit')
        btnEdit.setToolTip('Edit a Hidden Layer')
        btnEdit.clicked.connect(self.editLayer)
        btnRemove = QPushButton('Remove')
        btnRemove.setToolTip('Remove a Hidden Layer')
        btnRemove.clicked.connect(self.removeLayer)
        btnLoad = QPushButton('Load')
        btnLoad.setToolTip('Load a NN File')
        btnLoad.clicked.connect(self.showNNFileDialog)
        btnSave = QPushButton('Save')
        btnSave.setToolTip('Save a NN File')
        btnSave.clicked.connect(self.saveNNFileDialog)
        btnMakeDefault = QPushButton('Make default')
        btnMakeDefault.setToolTip('Make this as a default NN layer')
        btnMakeDefault.clicked.connect(self.makeDefaultNN)

        layout.addWidget(self.tableNNWidget, 0, 0, 9, 6)
        layout.addWidget(btnAdd, 9, 0)
        layout.addWidget(btnEdit, 9, 1)
        layout.addWidget(btnRemove, 9, 2)
        layout.addWidget(btnLoad, 9, 3)
        layout.addWidget(btnSave, 9, 4)
        layout.addWidget(btnMakeDefault, 9, 5)

        return layout

    def initMLOption(self):
        layout = QGridLayout()

        batchLabel = QLabel('Batch Size')
        batchLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editBatch = QLineEdit('320')
        self.editBatch.setFixedWidth(100)
        epochLabel = QLabel('Epoch')
        epochLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editEpoch = QLineEdit('20')
        self.editEpoch.setFixedWidth(100)
        lrLabel = QLabel('Learning Rate')
        self.editLR = QLineEdit('0.0005')
        self.editLR.setFixedWidth(100)
        self.cbVerbose = QCheckBox('Verbose')
        # self.cbVerbose.setChecked(True)

        splitLabel = QLabel('Split for Validation (0 means no split-data for validation)')
        self.editSplit = QLineEdit('0')
        self.editSplit.setFixedWidth(100)
        self.cbEarlyStop = QCheckBox('Use Early Stopping (validation data)')

        self.cbMinMax = QCheckBox('Use Min/Max of  All data')
        self.cbValidData = QCheckBox('Use Partially checked  sensors as validation')
        self.cbValidData.stateChanged.connect(self.validDataClicked)
        self.epochPbar = QProgressBar()

        layout.addWidget(batchLabel, 0, 0, 1, 1)
        layout.addWidget(self.editBatch, 0, 1, 1, 1)
        layout.addWidget(epochLabel, 0, 2, 1, 1)
        layout.addWidget(self.editEpoch, 0, 3, 1, 1)
        layout.addWidget(lrLabel, 0, 4, 1, 1)
        layout.addWidget(self.editLR, 0, 5, 1, 1)
        layout.addWidget(self.cbVerbose, 0, 6, 1, 1)

        layout.addWidget(splitLabel, 1, 0, 1, 2)
        layout.addWidget(self.editSplit, 1, 2, 1, 1)
        layout.addWidget(self.cbEarlyStop, 1, 3, 1, 2)

        layout.addWidget(self.cbMinMax, 2, 0, 1, 2)
        layout.addWidget(self.cbValidData, 2, 2, 1, 2)
        layout.addWidget(self.epochPbar, 2, 4, 1, 4)

        return layout

    def initCommand(self):
        layout = QGridLayout()

        outlierBtn = QPushButton('Remove Outlier')
        outlierBtn.clicked.connect(self.removeOutlier)
        correctTrailBtn = QPushButton('Correct Trail')
        correctTrailBtn.clicked.connect(self.correctTrail)
        fitShowBtn = QPushButton('Smooth Fit')
        fitShowBtn.clicked.connect(self.fitAndShow)
        fitFriedlander = QPushButton('Friedlander Fit')
        fitFriedlander.clicked.connect(self.fitFriedlander)
        saveDataBtn = QPushButton('Save Data')
        saveDataBtn.clicked.connect(self.saveDataAll)

        mlWithDataBtn = QPushButton('ML with Data')
        mlWithDataBtn.clicked.connect(self.doMachineLearningWithData)
        self.cbResume = QCheckBox('Resume Learning')
        self.cbResume.setChecked(True)
        saveModelBtn = QPushButton('Save Model')
        saveModelBtn.clicked.connect(self.saveModel)
        loadModelBtn =  QPushButton('Load Model')
        loadModelBtn.clicked.connect(self.loadModel)
        checkValBtn = QPushButton('Check Trained')
        checkValBtn.clicked.connect(self.checkVal)

        layout.addWidget(outlierBtn, 0, 0, 1, 1)
        layout.addWidget(correctTrailBtn, 0, 1, 1, 1)
        layout.addWidget(fitShowBtn, 0, 2, 1, 1)
        layout.addWidget(fitFriedlander, 0, 3, 1, 1)
        layout.addWidget(saveDataBtn, 0, 4, 1, 1)
        layout.addWidget(mlWithDataBtn, 1, 0, 1, 1)
        layout.addWidget(self.cbResume, 1, 1, 1, 1)
        layout.addWidget(saveModelBtn, 1, 2, 1, 1)
        layout.addWidget(loadModelBtn, 1, 3, 1, 1)
        layout.addWidget(checkValBtn, 1, 4, 1, 1)

        return layout

    def initGridTable(self):
        layout = QGridLayout()

        self.tableGridWidget = QTableWidget()

        self.tableGridWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # self.tableGridWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.tableGridWidget.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.tableGridWidget.setColumnCount(1)
        item = QTableWidgetItem('')
        self.tableGridWidget.setHorizontalHeaderItem(0, item)

        # Buttons
        btnAddDist = QPushButton('Add Distance')
        btnAddDist.setToolTip('Add a Distance')
        btnAddDist.clicked.connect(self.addDistance)
        btnAddHeight = QPushButton('Add Height')
        btnAddHeight.setToolTip('Add a Height')
        btnAddHeight.clicked.connect(self.addHeight)
        btnRemoveDist = QPushButton('Remove Distance')
        btnRemoveDist.setToolTip('Remove a Distance')
        btnRemoveDist.clicked.connect(self.removeDistance)
        btnRemoveHeight = QPushButton('Remove Height')
        btnRemoveHeight.setToolTip('Remove a Height')
        btnRemoveHeight.clicked.connect(self.removeHeight)
        btnLoadDistHeight = QPushButton('Load')
        btnLoadDistHeight.setToolTip('Load predefined Distance/Height structure')
        btnLoadDistHeight.clicked.connect(self.loadDistHeight)
        predictBtn = QPushButton('Predict')
        predictBtn.clicked.connect(self.predict)

        layout.addWidget(self.tableGridWidget, 0, 0, 9, 6)
        layout.addWidget(btnAddDist, 9, 0)
        layout.addWidget(btnAddHeight, 9, 1)
        layout.addWidget(btnRemoveDist, 9, 2)
        layout.addWidget(btnRemoveHeight, 9, 3)
        layout.addWidget(btnLoadDistHeight, 9, 4)
        layout.addWidget(predictBtn,9, 5)

        return layout

    def initUI(self):
        self.setWindowTitle('Machine Learning Curve Fitting/Interpolation')
        self.setWindowIcon(QIcon('web.png'))

        self.initMenu()

        layout = QVBoxLayout()

        sensorLayout = self.initSensor()
        readOptLayout = self.initReadOption()
        fileLayout = self.initCSVFileReader()
        cmdLayout = self.initCommand()
        nnLayout = self.initNNTable()
        mlOptLayout = self.initMLOption()
        tableLayout = self.initGridTable()

        layout.addLayout(fileLayout)
        layout.addLayout(sensorLayout)
        layout.addLayout(readOptLayout)
        layout.addLayout(mlOptLayout)
        layout.addLayout(nnLayout)
        layout.addLayout(cmdLayout)
        layout.addLayout(tableLayout)

        self.centralWidget().setLayout(layout)

        self.resize(900, 800)
        self.center()
        self.show()

    def showNNFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open NN file', './', filter="NN file (*.nn);;All files (*)")
        if fname[0] != '':
            self.updateNNList(fname[0])

    def updateNNList(self, filename):
        self.tableNNWidget.setRowCount(0)
        with open(filename, "r") as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                strAr = line.split(',')
                self.tableNNWidget.insertRow(count)
                self.tableNNWidget.setItem(count, 0, QTableWidgetItem(strAr[0]))
                self.tableNNWidget.setItem(count, 1, QTableWidgetItem(strAr[1].rstrip()))
                count += 1

    def saveNNFileDialog(self):
        fname = QFileDialog.getSaveFileName(self, 'Save NN file', './', filter="NN file (*.nn)")
        if fname[0] != '':
            self.saveNNFile(fname[0])

    def makeDefaultNN(self):
        filename = 'default.nn'
        self.saveNNFile(filename)
        QMessageBox.information(self, 'Saved', 'Neural Network Default Layers are set')

    def saveNNFile(self, filename):
        with open(filename, "w") as f:
            count = self.tableNNWidget.rowCount()
            for row in range(count):
                unit = self.tableNNWidget.item(row, 0).text()
                af = self.tableNNWidget.item(row, 1).text()
                f.write(unit + "," + af + "\n")

    def getNNLayer(self):
        nnList = []
        count = self.tableNNWidget.rowCount()
        for row in range(count):
            unit = int(self.tableNNWidget.item(row, 0).text())
            af = self.tableNNWidget.item(row, 1).text()
            nnList.append((unit, af))

        return nnList

    def addLayer(self):
        dlg = LayerDlg()
        rc = dlg.exec()
        if rc == 1: # ok
            unit = dlg.tbUnits.text()
            af = dlg.cbActivation.currentText()
            size = self.tableNNWidget.rowCount()
            self.tableNNWidget.insertRow(size-1)
            self.tableNNWidget.setItem(size-1, 0, QTableWidgetItem(unit))
            self.tableNNWidget.setItem(size-1, 1, QTableWidgetItem(af))

    def editLayer(self):
        row = self.tableNNWidget.currentRow()
        if row == -1 or row == (self.tableNNWidget.rowCount() - 1):
            return

        unit = self.tableNNWidget.item(row, 0).text()
        af = self.tableNNWidget.item(row, 1).text()
        dlg = LayerDlg(unit, af)
        rc = dlg.exec()
        if rc == 1: # ok
            unit = dlg.tbUnits.text()
            af = dlg.cbActivation.currentText()
            self.tableNNWidget.setItem(row, 0, QTableWidgetItem(unit))
            self.tableNNWidget.setItem(row, 1, QTableWidgetItem(af))

    def removeLayer(self):
        row = self.tableNNWidget.currentRow()
        if row > 0 and row < (self.tableNNWidget.rowCount() - 1):
            self.tableNNWidget.removeRow(row)

    def addDistance(self):
        sDist, ok = QInputDialog.getText(self, 'Input Distance', 'Distance to add:')
        if ok:
            cc = self.tableGridWidget.columnCount()
            self.tableGridWidget.setColumnCount(cc + 1)
            item = QTableWidgetItem('S' + sDist + 'm')
            self.tableGridWidget.setHorizontalHeaderItem(cc, item)

    def addHeight(self):
        sHeight, ok = QInputDialog.getText(self, 'Input Height', 'Height to add:')
        if ok:
            rc = self.tableGridWidget.rowCount()
            self.tableGridWidget.setRowCount(rc + 1)
            item = QTableWidgetItem('H' + sHeight + 'm')
            self.tableGridWidget.setVerticalHeaderItem(rc, item)

    def removeDistance(self):
        col = self.tableGridWidget.currentColumn()
        if col == -1:
            QMessageBox.warning(self, 'Warning', 'Select any cell')
            return
        if col == 0:
            QMessageBox.warning(self, 'Warning', 'First column cannot be removed')
            return

        self.tableGridWidget.removeColumn(col)

    def removeHeight(self):
        row = self.tableGridWidget.currentRow()
        if row == -1:
            QMessageBox.warning(self, 'Warning', 'Select any cell')
            return
        self.tableGridWidget.removeRow(row)

    def loadDistHeight(self):
        fname = QFileDialog.getOpenFileName(self, 'Open distance/height data file', '/srv/MLData',
                                            filter="CSV file (*.csv);;All files (*)")
        with open(fname[0], "r") as f:
            self.tableGridWidget.setRowCount(0)
            self.tableGridWidget.setColumnCount(0)

            lines = f.readlines()

            distAr = lines[0].split(',')
            for sDist in distAr:
                cc = self.tableGridWidget.columnCount()
                self.tableGridWidget.setColumnCount(cc + 1)
                item = QTableWidgetItem('S' + sDist + 'm')
                self.tableGridWidget.setHorizontalHeaderItem(cc, item)

            heightAr = lines[1].split(',')
            for sHeight in heightAr:
                rc = self.tableGridWidget.rowCount()
                self.tableGridWidget.setRowCount(rc + 1)
                item = QTableWidgetItem('H' + sHeight + 'm')
                self.tableGridWidget.setVerticalHeaderItem(rc, item)

    def distClicked(self, state):
        senderName = self.sender().text()
        col = self.distArrayName.index(senderName)
        rows = len(self.cbArray)
        for i in range(rows):
            qcheckbox = self.cbArray[i][col]
            if state == Qt.Checked:
                qcheckbox.setChecked(True)
            else:
                qcheckbox.setChecked(False)

    def heightClicked(self, state):
        senderName = self.sender().text()
        row = self.heightArrayName.index(senderName)
        cols = len(self.cbArray[0])
        for i in range(cols):
            qcheckbox = self.cbArray[row][i]
            if state == Qt.Checked:
                qcheckbox.setChecked(True)
            else:
                qcheckbox.setChecked(False)

    def validDataClicked(self, state):
        if state == Qt.Checked:
            self.editSplit.setEnabled(False)
        else:
            self.editSplit.setEnabled(True)

    def showFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open data file', '/srv/MLData', filter="CSV file (*.csv);;All files (*)")
        if fname[0]:
            self.editFile.setText(fname[0])
            self.df = pd.read_csv(fname[0], dtype=float)

    def correctTrail(self):
        if self.dataLoaded == False:
            QMessageBox.information(self, 'Warning', 'Load Data First')
            return

        numSensors = len(self.southSensors)
        for i in range(numSensors):
            t_data = self.time_data
            s_data = self.southSensors[i]

            index_ij = self.indexijs[i]
            i = index_ij[0]
            j = index_ij[1]
            sensorName = self.distArrayName[j] + self.heightArrayName[i]

            self._correctTrail(t_data, s_data)

    def _correctTrail(self, xdata, ydata):
        data_f = np.copy(ydata)
        # for step in range(iterNum):
        data_f, index_at_max, overpressure, index_at_zero, impulse = self.doFriedlander(data_f)

        ydata3 = np.zeros(shape=(len(ydata),))
        for i in range(len(ydata)):
            if i <= index_at_zero:
                ydata3[i] = ydata[i] # original data
            else:
                ydata3[i] = 0


        plt.figure()
        rawLabel = 'original'
        plt.scatter(xdata, ydata, label=rawLabel, color="red", s=1)
        filterLabel2 = 'corrected'
        plt.scatter(xdata, ydata3, label=filterLabel2, color="green", s=1)
        plt.title('Fit')
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

        # change data for save
        for i in range(len(ydata)):
            ydata[i] = ydata3[i]

    def removeOutlier(self):
        if self.dataLoaded == False:
            QMessageBox.information(self, 'Warning', 'Load Data First')
            return

        checkEveryIter = self.cbOutIter.isChecked()
        tolerance = float(self.editOutlier.text())

        iterNum = int(self.editIteration.text())
        numSensors = len(self.southSensors)
        MAX_ITERATION = 30
        for i in range(numSensors):
            t_data = self.time_data
            s_data = self.southSensors[i]
            ori_data = np.copy(s_data)

            self.iterPbar.setMaximum(iterNum)

            if self.cbOutLoop.isChecked():
                noLoop = 0
                while True:
                    data_f = np.copy(s_data)
                    for step in range(iterNum):
                        data_f = self.doSmoothing(data_f, True)
                        self.iterPbar.setValue(step + 1)

                    noOfOutliers = self.examinOutliers(t_data, s_data, data_f, tolerance, False)
                    noLoop += 1
                    if noOfOutliers == 0 or noLoop > MAX_ITERATION:
                        fig, axs = plt.subplots(2, 1, figsize=(12, 12))
                        title = 'Original vs Outliers removed'
                        fig.suptitle(title)

                        axs[0].scatter(t_data, ori_data, label='original data', color="red", s=1)
                        axs[0].grid()

                        axs[1].scatter(t_data, s_data, label='data with outliers removed', color="red", s=1)
                        axs[1].grid()

                        plt.show()
                        break
            else:
                data_f = np.copy(s_data)

                for step in range(iterNum):
                    data_f = self.doSmoothing(data_f, True)

                    self.iterPbar.setValue(step + 1)

                self.examinOutliers(t_data, s_data, data_f, tolerance, True)

        if self.cbOutLoop.isChecked():
            reply = QMessageBox.question(self, 'Message', 'Removing Outliers Done. Do you want to save it?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.saveDataAll()

    def examinOutliers(self, time_data, data, data_f, outlierTolerance, showGraph):
        outx = []
        outliers = []
        size = len(data)
        dmin = data_f.min()
        dmax = data_f.max()
        tolerance = (dmax - dmin) * outlierTolerance / 100.0

        for i in range(size):
            raw = data[i]
            filtered = data_f[i]
            if filtered - raw > tolerance: # or abs(filtered - raw) > tolerance:
                outx.append(time_data[i])
                outliers.append(raw)

                # swap values
                data[i] = filtered
                raw = filtered

        noOfOutliers = len(outx)

        if showGraph:
            fig, axs = plt.subplots(2, 1, figsize=(12, 12))
            title = 'Outliers (' + str(noOfOutliers) + ' swapped)'
            fig.suptitle(title)

            axs[0].scatter(time_data, data, label='data', color="red", s=1)
            axs[0].scatter(outx, outliers, label='outlier', color="blue", s=1)

            axs[0].grid()

            axs[1].scatter(time_data, data, label='data', color="red", s=1)
            axs[1].grid()

            plt.show()

        return noOfOutliers

    def fitAndShow(self):
        if self.dataLoaded == False:
            QMessageBox.information(self, 'Warning', 'Load Data First')
            return

        iterNum = int(self.editIteration.text())
        self.iterPbar.setMaximum(iterNum)
        numSensors = len(self.southSensors)
        for i in range(numSensors):
            t_data = self.time_data
            s_data = self.southSensors[i]

            data_f = np.copy(s_data)
            for step in range(iterNum):
                data_f = self.doSmoothing(data_f, True)
                self.iterPbar.setValue(step + 1)

            data_label = self.cbSmoothing.currentText()
            selectedSmoothing = self.cbSmoothing.currentIndex()
            if selectedSmoothing > 0:
                data_label += '/' + self.editWin.text()
            if selectedSmoothing > 1:
                data_label += '/' + self.editOrder.text()

            index_ij = self.indexijs[i]
            i = index_ij[0]
            j = index_ij[1]
            sensorName = self.distArrayName[j] + self.heightArrayName[i]

            self.checkDataGraph(sensorName, t_data, s_data, data_f, data_label, iterNum)

            # finally, replace s_data for save
            for j in range(len(s_data)):
                s_data[j] = data_f[j]

    def fitFriedlander(self):
        if self.dataLoaded == False:
            QMessageBox.information(self, 'Warning', 'Load Data First')
            return

        # iterNum = int(self.editIteration.text())
        numSensors = len(self.southSensors)
        self.iterPbar.setMaximum(numSensors)

        resultArray = []
        for i in range(numSensors):
            t_data = self.time_data
            s_data = self.southSensors[i]

            data_f = np.copy(s_data)
            # for step in range(iterNum):
            data_f, index_at_max, overpressure, index_at_zero, impulse = self.doFriedlander(data_f)
            self.iterPbar.setValue(i)

            data_label = 'Friedlander'

            index_ij = self.indexijs[i]
            ii = index_ij[0]
            j = index_ij[1]
            sensorName = self.distArrayName[j] + self.heightArrayName[ii]

            self.checkDataGraph2(sensorName, t_data, s_data, data_f, data_label, index_at_max, overpressure, index_at_zero)

            one_line = str(self.distList[j]) + ',' + str(self.heightList[ii]) + ',' + str(index_at_max) + ',' + str(overpressure) + ',' + str(index_at_zero) + ',' + str(impulse)
            resultArray.append(one_line)

            # finally, replace s_data for save
            for j in range(len(s_data)):
                s_data[j] = data_f[j]

        reply = QMessageBox.question(self, 'Message', 'Do you want to save index_at_max, overpressure, index_at_zero to a file?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            suggestion = '/srv/MLData/indexOverpressure.csv'
            filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")

            if filename[0] != '':
                file = open(filename[0], 'w')

                column1 = 'Distance,Height,index_max,Overpressure,index_zero,impulse\n'
                file.write(column1)

                for col in resultArray:
                    file.write(col + '\n')

    def loadData(self):
        filename = self.editFile.text()
        if filename == '':
            QMessageBox.about(self, 'Warining', 'No CSV Data')
            return

        startRowNum = int(self.startRow.text())
        endRowNum = int(self.endRow.text())
        if startRowNum >= endRowNum:
            QMessageBox.about(self, 'Warining', 'startRow is bigger than endRow')
            QApplication.restoreOverrideCursor()
            return
        stepNum = int(self.step.text())
        if stepNum <= 0:
            QMessageBox.about(self, 'Warining', 'stepNum should be bigger than 0')
            QApplication.restoreOverrideCursor()
            return

        rows = len(self.heightArrayName)
        cols = len(self.distArrayName)

        listSelected = []
        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray[i][j]
                if qcheckbox.checkState() != Qt.Unchecked:
                    listSelected.append((i, j))

        if len(listSelected) == 0:
            QMessageBox.information(self, 'Warning', 'Select sensor(s) first..')
            return

        try:
            self.readCSV(listSelected, startRowNum, endRowNum, stepNum)

        except ValueError:
            QMessageBox.information(self, 'Error', 'There is some error...')
            return

        self.dataLoaded = True
        QMessageBox.information(self, 'Done', 'Data is Loaded')

    def readCSV(self, listSelected, startRowNum, endRowNum, stepNum):
        self.indexijs.clear()
        self.time_data = None
        self.southSensors.clear()

        self.time_data = self.df.values[startRowNum:endRowNum:stepNum, 1:2].flatten()

        for index_ij in listSelected:
            sensorName, data = self.getPressureData(index_ij, startRowNum, endRowNum, stepNum)
            self.indexijs.append(index_ij)
            self.southSensors.append(data)

    def showGraphs(self):
        if self.dataLoaded == False:
            QMessageBox.information(self, 'Warning', 'Load Data First')
            return

        numSensors = len(self.southSensors)

        plt.figure()
        for i in range(numSensors):
            t_data = self.time_data
            s_data = self.southSensors[i]

            max_val = max(s_data)
            smax_val = format(max_val, '.2f')
            index_at_max = max(range(len(s_data)), key=s_data.__getitem__)
            impulse = self.getImpulse(s_data)
            simpulse = format(impulse, '.2f')
            index_at_zero = index_at_max
            for index in range(index_at_max, len(s_data)-1):
                val = s_data[index]
                val_n = s_data[index+1]
                if val >=0 and val_n <=0:
                    index_at_zero = index
                    break
            time_at_zero = t_data[index_at_zero]
            stime_at_zero = format(time_at_zero, '.6f')

            index_ij = self.indexijs[i]
            i = index_ij[0]
            j = index_ij[1]
            sensorName = self.distArrayName[j] + self.heightArrayName[i] + ' (max:' + smax_val + ')'
            # sensorName = self.distArrayName[j] + self.heightArrayName[
            #     i] + ' (max:' + smax_val + ',impulse:' + simpulse + ',timeAtZero=' + stime_at_zero + ')'

            plt.scatter(t_data, s_data, label=sensorName, s=1)

        plt.title('Pressure Graph')
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right', markerscale=4.)
        plt.grid()

        plt.show()

    def getPressureData(self, index_ij, startRowNum, endRowNum, stepNum):

        i = index_ij[0]
        j = index_ij[1]
        col_no = (i+2) + (j * len(self.heightArray))
        sensorName = self.distArrayName[j] + self.heightArrayName[i]

        data_raw = self.df.values[::, col_no:col_no+1].flatten()

        index_at_max = max(range(len(data_raw)), key=data_raw.__getitem__)
        newStartNum = startRowNum
        newEndNum = endRowNum
        if stepNum != 1:
            newStartNum = index_at_max
            while newStartNum > startRowNum:
                newStartNum -= stepNum
            newStartNum += stepNum
            newEndNum = index_at_max
            while newEndNum < endRowNum:
                newEndNum += stepNum
            newEndNum -= stepNum

            data = self.df.values[newStartNum:newEndNum:stepNum, col_no:col_no+1].flatten()
        else:
            data = self.df.values[startRowNum:endRowNum:stepNum, col_no:col_no+1].flatten()

        return sensorName, data

    def getMinMaxPresssureOfAllData(self):
        maxPressure = -100000
        minPressure = 100000
        for i in range(len(self.distList)*len(self.heightList)):
            one_data = self.df.values[::, i+2:i+3].flatten()
            maxp_local = max(one_data)
            minp_local = min(one_data)
            if maxp_local > maxPressure:
                maxPressure = maxp_local
            if minp_local < minPressure:
                minPressure = minp_local

        return maxPressure, minPressure

    def getMinMaxPressureOfLoadedData(self):
        maxPressure = -100000
        minPressure = 100000

        numSensors = len(self.southSensors)
        for i in range(numSensors):
            s_data = self.southSensors[i]
            maxp_local = max(s_data)
            minp_local = min(s_data)
            if maxp_local > maxPressure:
                maxPressure = maxp_local
            if minp_local < minPressure:
                minPressure = minp_local

        return maxPressure, minPressure

    def doSmoothing(self, data, showGraph):
        QApplication.setOverrideCursor(Qt.WaitCursor)

        selectedSmoothing = self.cbSmoothing.currentIndex()
        smoothingWindow = int(self.editWin.text())
        smoothingOrder = int(self.editOrder.text())

        retainPeak = self.cbPreservePeak.isChecked()
        peakWindow = int(self.editPeakWindow.text())

        if selectedSmoothing == 1:
            if smoothingWindow % 2 == 0:
                QMessageBox.about(self, 'Warining', 'smoothingWindow needs to be odd number')
                raise ValueError('Error')

        elif selectedSmoothing == 2:
            if smoothingWindow % 2 == 0 or smoothingOrder >= smoothingWindow:
                QMessageBox.about(self, 'Warining',
                                  'smoothingWindow needs to be odd number and bigger than smoothingOrder')
                raise ValueError('Error')

        data_f = np.copy(data)
        smoothFilter = SmoothFilter(selectedSmoothing)
        data_f = smoothFilter.getFilteredData(data_f, smoothingWindow, smoothingOrder, retainPeak, peakWindow)

        QApplication.restoreOverrideCursor()

        return data_f

    def doFriedlander(self, data):
        QApplication.setOverrideCursor(Qt.WaitCursor)

        data_f = np.copy(data)

        index_at_max = max(range(len(data)), key=data.__getitem__)
        overpressure = max(data)

        sumImpulse = 0
        impulseArray = []
        for i in range(len(data_f)):
            cur_p = data_f[i]
            sumImpulse += cur_p
            impulseArray.append(sumImpulse)

        index_at_zero = max(range(len(impulseArray)), key=impulseArray.__getitem__)
        impulse = max(impulseArray)

        index_at_zero_from_peak = index_at_zero - index_at_max

        tindex = 0
        for i in range(len(data)):
            if i < index_at_max:
                data_f[i] = 0
            else:
                data_f[i] = overpressure * math.exp(-1 * tindex / index_at_zero_from_peak) * (1 - tindex / index_at_zero_from_peak)
                tindex += 1

        QApplication.restoreOverrideCursor()

        return data_f, index_at_max, overpressure, index_at_zero, impulse

    def doMachineLearningWithData(self):
        if not self.indexijs or not self.southSensors:
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        batchSize = int(self.editBatch.text())
        epoch = int(self.editEpoch.text())
        if epoch < 1:
            QMessageBox.warning(self, 'warning', 'Epoch shall be greater than 0')
            return

        learningRate = float(self.editLR.text())
        verbose = self.cbVerbose.isChecked()

        # useValidation = self.cbValidData.isChecked()
        splitPercentage = float(self.editSplit.text())
        earlyStopping = self.cbEarlyStop.isChecked()

        x_train_data, y_train_data, x_valid_data, y_valid_data = self._prepareForMachineLearning()
        if len(x_valid_data) == 0:
            self.doMachineLearning(x_train_data, y_train_data, batchSize, epoch, learningRate, splitPercentage,
                                   earlyStopping, verbose)
        else:
            self.doMachineLearningWithValidation(x_train_data, y_train_data, x_valid_data, y_valid_data, batchSize,
                                                 epoch, learningRate, splitPercentage, earlyStopping, verbose)

        QApplication.restoreOverrideCursor()

    def saveData(self):
        if not self.indexijs or not self.time_data or not self.southSensors:
            QMessageBox.about(self, 'Warining', 'No Data is prepared')
            return

        numSensors = len(self.southSensors)
        for i in range(numSensors):
            index_ij = self.indexijs[i]
            t_data = self.time_data
            s_data = self.southSensors[i]

            suggestion = '/srv/MLData/' + self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]] + '.csv'
            filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")

            if filename[0] != '':
                file = open(filename[0], 'w')

                column1 = 'Time, ' + self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]] + '\n'
                file.write(column1)

                numData = len(s_data)
                for j in range(numData):
                    line = str(t_data[j]) + ',' + str(s_data[j])
                    file.write(line)
                    file.write('\n')

                file.close()

                QMessageBox.information(self, "Save", filename[0] + " is saved successfully")

    def saveDataAll(self):
        suggestion = '/srv/MLData/Changed.csv'
        # suggestion = '../MLData/' + self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]] + '.csv'
        filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")
        if filename[0] != '':
            file = open(filename[0], 'w')

            column1 = 'Time, '
            numSensors = len(self.southSensors)
            for i in range(numSensors):
                index_ij = self.indexijs[i]
                column1 += self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]]
                if i != (numSensors - 1):
                    column1 += ','
            column1 += '\n'

            file.write(column1)

            t_data = self.time_data

            numData = len(t_data)
            for j in range(numData):
                line = str(t_data[j]) + ','

                for i in range(numSensors):
                    s_data = self.southSensors[i]
                    line += str(s_data[j])
                    if i != (numSensors - 1):
                        line += ','

                file.write(line)
                file.write('\n')

            file.close()

            QMessageBox.information(self, "Save", filename[0] + " is saved successfully")

    def saveModel(self):
        suggestion = '/srv/MLData/tfModel'
        filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "Model")
        if filename[0] != '':
            self.modelLearner.saveModel(filename[0])

        QMessageBox.information(self, 'Saved', 'Model is saved.')

    def loadModel(self):
        folderpath = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folderpath != '':
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.modelLearner.loadModel(folderpath)
            QApplication.restoreOverrideCursor()

        QMessageBox.information(self, 'Loaded', 'Model is loaded.')

    def checkVal(self):
        if not self.indexijs or not self.southSensors:
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return
        if self.modelLearner.modelLoaded == False:
            QMessageBox.about(self, 'Warning', 'Model is not created/loaded')
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        x_data, y_data, x_valid_data, y_valid_data = self._prepareForMachineLearning()
        y_predicted = self.modelLearner.predict(x_data)

        QApplication.restoreOverrideCursor()

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        plt.figure()
        plt.scatter(x_display, y_data, label='original data', color="red", s=1)
        plt.scatter(x_display, y_predicted, label='predicted', color="blue", s=1)
        plt.title('Machine Learning Prediction')
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def predict(self):
        if self.time_data is None:
            QMessageBox.warning(self, 'Warning', 'For timedata, load at least one data')
            return

        if self.modelLearner.modelLoaded == False:
            QMessageBox.about(self, 'Warning', 'Model is not created/loaded')
            return

        distCount = self.tableGridWidget.columnCount()
        heightCount = self.tableGridWidget.rowCount()
        if distCount < 1 or heightCount < 1:
            QMessageBox.warning(self, 'Warning', 'You need to add distance or height to predict')
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        sDistHeightArray = []
        distHeightArray = []

        for i in range(distCount):
            dist = self.tableGridWidget.horizontalHeaderItem(i).text()
            distOnly = dist[1:len(dist)-1]

            for j in range(heightCount):
                height = self.tableGridWidget.verticalHeaderItem(j).text()
                heightOnly = height[1:len(height)-1]

                distf = float(distOnly)
                heightf = float(heightOnly)

                sDistHeightArray.append(dist+height)
                distHeightArray.append((distf, heightf))

        # numSensors = len(distHeightArray)
        # if self.cbMinMax.isChecked() == True:
        #     maxp, minp = self.getMinMaxPresssureOfAllData()
        # else:
        #     maxp, minp = self.getMinMaxPressureOfLoadedData()

        distList_n = np.copy(self.distList)
        maxDist = max(distList_n)
        minDist = min(distList_n)

        heightList_n = np.copy(self.heightList)
        maxHeight = max(heightList_n)
        minHeight = min(heightList_n)

        y_array = []
        for distHeight in distHeightArray:
            x_data = self._prepareOneSensorForPredict(distHeight, maxDist, minDist, maxHeight, minHeight)

            y_predicted = self.modelLearner.predict(x_data)
            # self.unnormalize(y_predicted, maxp, minp)

            y_array.append(y_predicted)

        QApplication.restoreOverrideCursor()

        resultArray = self.showPredictionGraphs(sDistHeightArray, distHeightArray, y_array)

        reply = QMessageBox.question(self, 'Message', 'Do you want to save overpressure and impulse to a file?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            suggestion = '/srv/MLData/opAndImpulse.csv'
            filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")

            if filename[0] != '':
                file = open(filename[0], 'w')

                column1 = 'distance,height,indexAtMax,overpressure,indexAtZero,impulse\n'
                file.write(column1)

                for col in resultArray:
                    file.write(col+'\n')

    def unnormalize(self, data, max, min):
        for i in range(len(data)):
            data[i] = data[i] * (max - min) + min

    def _prepareOneSensorForPredict(self, distHeight, maxDist, minDist, maxHeight, minHeight):
        distance = (distHeight[0] - minDist) / (maxDist - minDist)
        height = (distHeight[1] - minHeight) / (maxHeight - minHeight)

        NUM_DATA = 100000
        feature_size = 3  # pressure and height and distance
        x_data = np.zeros(shape=(NUM_DATA, feature_size))

        for i in range(NUM_DATA):
            x_data[i][0] = self.time_data[i]
            x_data[i][1] = height
            x_data[i][2] = distance

        return x_data

    def showPredictionGraphs(self, sDistHeightArray, distHeightArray, y_array):
        # numSensors = len(y_array)
        resultArray = []

        plt.figure()
        for i in range(len(y_array)):
            t_data = self.time_data
            s_data = y_array[i]

            distHeight = distHeightArray[i]
            lab = sDistHeightArray[i]

            distance = distHeight[0]
            height = distHeight[1]

            index_at_max = max(range(len(s_data)), key=s_data.__getitem__)
            overpressure = max(s_data)
            impulse, index_at_zero = self.getImpulseAndIndexZero(s_data)

            dispLabel = lab + '/op=' + format(overpressure[0], '.2f') + '/impulse=' + format(impulse, '.2f')

            resultArray.append(str(distance) + ',' + str(height) + ',' + str(index_at_max) + ',' +
                               format(overpressure[0], '.6f') + ',' + str(index_at_zero) + ',' + format(impulse, '.6f'))

            plt.scatter(t_data, s_data, label=dispLabel, s=1)

        plt.title('Pressure Graph')
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right', markerscale=4.)
        plt.grid()

        plt.show()

        return resultArray

    def getImpulse(self, data):
        sumImpulse = 0
        impulseArray = []
        for i in range(len(data)):
            cur_p = data[i]
            sumImpulse += cur_p
            impulseArray.append(sumImpulse)

        impulse = max(impulseArray)

        return impulse

    def getImpulseAndIndexZero(self, data):
        index_at_max = max(range(len(data)), key=data.__getitem__)

        sumImpulse = 0
        impulseArray = []
        initP = data[0][0]
        for i in range(len(data)):
            cur_p = data[i][0]
            if cur_p > 0 and cur_p <= initP :
                cur_p = 0

            sumImpulse += cur_p * 0.000002
            impulseArray.append(sumImpulse)

        # index_at_zero = max(range(len(impulseArray)), key=impulseArray.__getitem__)
        impulse = max(impulseArray)
        index_at_zero = impulseArray.index(impulse)

        return impulse, index_at_zero

    def checkDataGraph(self, sensorName, time_data, rawdata, filetered_data, data_label, iterNum):
        impulse_original = self.getImpulse(rawdata)
        impulse_filtered = self.getImpulse(filetered_data)

        plt.figure()

        rawLabel = 'Raw-Normalized (impulse=' + format(impulse_original, '.4f') + ')'
        plt.scatter(time_data, rawdata, label=rawLabel, color="red", s=1)
        filterLabel = data_label + ' (iter=' + str(iterNum) + ', impulse=' + format(impulse_filtered, '.4f') + ')'
        plt.scatter(time_data, filetered_data, label=filterLabel, color="blue", s=1)
        plt.title(sensorName)
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def checkDataGraph2(self, sensorName, time_data, rawdata, filetered_data, data_label, index_at_max, overpressure, index_at_zero):
        impulse_original = self.getImpulse(rawdata)
        impulse_filtered = self.getImpulse(filetered_data)

        plt.figure()

        rawLabel = 'Raw-Normalized (impulse=' + format(impulse_original, '.4f') + ')'
        plt.scatter(time_data, rawdata, label=rawLabel, color="red", s=1)
        filterLabel = data_label + ', impulse=' + format(impulse_filtered, '.4f') + ',indexMax=' + str(index_at_max) \
                      + ',Overpressure=' + str(overpressure) + ',indexZero=' + str(index_at_zero)
        plt.scatter(time_data, filetered_data, label=filterLabel, color="blue", s=1)
        plt.title(sensorName)
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def prepareOneSensorData(self, height, dist, data, maxp, minp):
        datasize = len(data)
        cur_data = list(data)

        # for j in range(datasize):
        #     cur_data[j] = (cur_data[j] - minp) / (maxp - minp)

        feature_size = 3  # pressure and height and distance
        x_data = np.zeros((datasize, feature_size))

        # fill the data
        for j in range(datasize):
            x_data[j][0] = self.time_data[j]
            x_data[j][1] = height
            x_data[j][2] = dist

        y_data = np.zeros((datasize, 1))
        for j in range(datasize):
            y_data[j][0] = cur_data[j]

        return x_data, y_data

    def _prepareForMachineLearning(self):
        heightList_n = np.copy(self.heightList)
        maxHeight = max(heightList_n)
        minHeight = min(heightList_n)
        for j in range(len(heightList_n)):
            heightList_n[j] = (heightList_n[j] - minHeight) / (maxHeight - minHeight)

        distList_n = np.copy(self.distList)
        maxDist = max(distList_n)
        minDist = min(distList_n)
        for j in range(len(distList_n)):
            distList_n[j] = (distList_n[j] - minDist) / (maxDist - minDist)

        NUM_DATA = 100000
        numSensors = len(self.southSensors)

        # separate data to train & validation
        trainArray = []
        trainIndex = []
        validArray = []
        validIndex = []

        useValidation = self.cbValidData.isChecked()
        if useValidation:
            for i in range(numSensors):
                indexij = self.indexijs[i]
                row_num = indexij[0]
                col_num = indexij[1]
                one_data = self.southSensors[i]

                if self.cbArray[row_num][col_num].checkState() == Qt.PartiallyChecked:
                    validArray.append(one_data)
                    validIndex.append(indexij)
                elif self.cbArray[row_num][col_num].checkState() == Qt.Checked:
                    trainArray.append(one_data)
                    trainIndex.append(indexij)
        else:
            for i in range(numSensors):
                indexij = self.indexijs[i]
                # row_num = indexij[0]
                one_data = self.southSensors[i]

                trainArray.append(one_data)
                trainIndex.append(indexij)

        if len(trainArray) == 0:
            QMessageBox.warning(self, 'warning', 'No Training Data!')
            return

        x_train_data = np.zeros(shape=(NUM_DATA * len(trainArray), 3))
        y_train_data = np.zeros(shape=(NUM_DATA * len(trainArray) , 1))

        x_valid_data = np.zeros(shape=(NUM_DATA * len(validArray), 3))
        y_valid_data = np.zeros(shape=(NUM_DATA * len(validArray), 1))

        # get maxp/minp from all of the training data
        maxp = -1000000
        minp = 100000
        if self.cbMinMax.isChecked() == True:
            maxp, minp = self.getMinMaxPresssureOfAllData()
        else:
            for i in range(len(trainArray)):
                one_data = list(trainArray[i])
                maxp_local = max(one_data)
                minp_local = min(one_data)
                if maxp_local > maxp:
                    maxp = maxp_local
                if minp_local < minp:
                    minp = minp_local

        for i in range(len(trainArray)):
            index_ij = trainIndex[i]
            s_data = trainArray[i]
            x_data_1, y_data_1 = self.prepareOneSensorData(heightList_n[index_ij[0]], distList_n[index_ij[1]], s_data,
                                                           maxp, minp)

            for j in range(NUM_DATA):
                x_train_data[i*NUM_DATA + j] = x_data_1[j]
                y_train_data[i*NUM_DATA + j] = y_data_1[j]

        for i in range(len(validArray)):
            index_ij = validIndex[i]
            s_data = validArray[i]
            x_data_1, y_data_1 = self.prepareOneSensorData(heightList_n[index_ij[0]], distList_n[index_ij[1]], s_data,
                                                           maxp, minp)

            for j in range(NUM_DATA):
                x_valid_data[i*NUM_DATA + j] = x_data_1[j]
                y_valid_data[i*NUM_DATA + j] = y_data_1[j]

        return x_train_data, y_train_data, x_valid_data, y_valid_data

    def doMachineLearning(self, x_data, y_data, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb):
        self.epochPbar.setMaximum(epoch)

        if self.cbResume.isChecked() == False or self.modelLearner.modelLoaded == False:
            nnList = self.getNNLayer()
            self.modelLearner.set(nnList, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb,
                                  TfCallback(self.epochPbar))

        training_history = self.modelLearner.fit(x_data, y_data)

        y_predicted = self.modelLearner.predict(x_data)
        self.modelLearner.showResult(y_data, training_history, y_predicted, 'Sensors', 'Height')

    def doMachineLearningWithValidation(self, x_train_data, y_train_data, x_valid_data, y_valid_data, batchSize, epoch,
                                        learningRate, splitPercentage, earlyStopping, verb):
        self.epochPbar.setMaximum(epoch)

        if self.cbResume.isChecked() == False or self.modelLearner.modelLoaded == False:
            nnList = self.getNNLayer()
            self.modelLearner.set(nnList, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb,
                                  TfCallback(self.epochPbar))

        training_history = self.modelLearner.fitWithValidation(x_train_data, y_train_data, x_valid_data, y_valid_data)

        y_predicted = self.modelLearner.predict(x_valid_data)
        self.modelLearner.showResult(y_valid_data, training_history, y_predicted, 'Sensors', 'Height')

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

class TfCallback(tf.keras.callbacks.Callback):
    def __init__(self, pbar):
        super().__init__()
        self.pbar = pbar
        self.curStep = 1

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.setValue(self.curStep)
        self.curStep += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MLWindow()
    sys.exit(app.exec_())