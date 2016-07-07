#include "correctionwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    CorrectionWindow w;
    w.show();

    return a.exec();
}
