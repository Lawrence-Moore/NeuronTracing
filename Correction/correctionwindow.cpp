#include "correctionwindow.h"
#include "ui_correctionwindow.h"

CorrectionWindow::CorrectionWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::CorrectionWindow)
{
    ui->setupUi(this);
}

CorrectionWindow::~CorrectionWindow()
{
    delete ui;
}
