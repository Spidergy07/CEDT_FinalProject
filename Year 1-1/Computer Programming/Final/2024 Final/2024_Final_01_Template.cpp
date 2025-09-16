#include <iostream>

using namespace std;

long long T(int n, int k) {  // Schroder Triangle


}
long long S(int n) {         // Schroder Number


}
long long W(int n) {         // Wedderburn–Etherington Number


}
int main() {
    string fn;
    int p1, p2;
    while (cin >> fn >> p1) {
        if (fn == "T") {
            cin >> p2;
            cout << fn << '(' << p1 << ',' << p2 << ") = " << T(p1,p2) << endl;
        } else if (fn == "S") {
            cout << fn << '(' << p1 << ") = " << S(p1) << endl;
        } else if (fn == "W") {
            cout << fn << '(' << p1 << ") = " << W(p1) << endl;
        }
    }
    return 0;
}