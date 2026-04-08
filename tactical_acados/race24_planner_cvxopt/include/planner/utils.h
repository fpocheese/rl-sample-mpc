#include<cmath>
namespace PlannerUtils {
    void lla2xyz(double &x, double &y, double &z, double lat, double lon, double hgt) {
        double A_EARTH = 6378137.0;
        double flattening = 1.0/298.257223563;
        double NAV_E2 = (2.0-flattening)*flattening;
        double deg2rad = M_PI/180.0;
        double slat = sin(lat*deg2rad);
        double clat = cos(lat*deg2rad);
        double r_n = A_EARTH/sqrt(1.0 - NAV_E2*slat*slat);
        x = (r_n + hgt)*clat*cos(lon*deg2rad);
        y = (r_n + hgt)*clat*sin(lon*deg2rad);
        z = (r_n*(1.0 - NAV_E2) + hgt)*slat;
    }

    void lla2xyz2Ref(double &x, double &y, double &z, double lat, double lon, double hgt) {
        double rtk_ref_point[3] = {24.4710399, 54.605548, 0.0};
        double refX, refY, refZ;
        lla2xyz(refX, refY, refZ, rtk_ref_point[0], rtk_ref_point[1], rtk_ref_point[2]);
        lla2xyz(x, y, z, lat, lon, hgt);
        x -= refX;
        y -= refY;
        z -= refZ;
    }

    void lla2xyz2Ref2d(double& x, double &y, double &z, double lat, double lon) {
        lla2xyz2Ref(x, y, z, lat, lon, 0.0);
    }

    void xyFLU2xyENU(double &xENU, double &yENU, double xFLU, double yFLU, double yawRadNED){
        double yawRadENU = M_PI/2.0 - yawRadNED;
        if (yawRadENU < 0.0){
            yawRadENU += 2.0*M_PI;
        }
        xENU = xFLU*sin(yawRadENU) - yFLU*cos(yawRadENU);
        yENU = xFLU*cos(yawRadENU) + yFLU*sin(yawRadENU);

        // xENU = xFLU*cos(yawRadENU) - yFLU*sin(yawRadENU);
        // yENU = xFLU*sin(yawRadENU) + yFLU*cos(yawRadENU);
    }
    // double convertEnuYawToLtpl(double enu_yaw) {
    //     double tmp = M_PI - enu_yaw ;
    //     return tmp;
    // }
    double convertEnuYawToLtpl(double enu_yaw) {
        double tmp = enu_yaw - M_PI / 2;
        if (tmp < -M_PI) {
            tmp += 2 * M_PI;
        }
        return tmp;
    }
}