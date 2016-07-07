#ifndef CORRECTIONWINDOW_H
#define CORRECTIONWINDOW_H

#include <QMainWindow>

namespace Ui {
class CorrectionWindow;
}

class CorrectionWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit CorrectionWindow(QWidget *parent = 0);
    ~CorrectionWindow();

private:
    Ui::CorrectionWindow *ui;
};

#endif // CORRECTIONWINDOW_H
