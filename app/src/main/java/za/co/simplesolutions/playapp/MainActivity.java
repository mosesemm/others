package za.co.simplesolutions.playapp;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ShareCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import java.util.Random;

public class MainActivity extends AppCompatActivity {

    private static final String LOG_TAG = MainActivity.class.getSimpleName();
    private String orderMessage = "";
    public static final String EXTRA_MESSAGE = "za.co.simplesolutions.playapp.extra.MESSAGE";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        Log.i(MainActivity.class.getName(), "started...");
    }

    public void onClick(View view) {
        /**Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                .setAction("Action", null).show();
         **/

        Intent intent = new Intent(MainActivity.this, OrderActivity.class);
        intent.putExtra(EXTRA_MESSAGE, orderMessage);
        startActivity(intent);
    }

    public void displayToast(String message)
    {
        Toast.makeText(getApplicationContext(), message, Toast.LENGTH_SHORT).show();
    }

    public void showDonutOrder(View view) {
        orderMessage = getString(R.string.donut_order_message);
        displayToast(getString(R.string.donut_order_message));
    }

    public void showIcreamOrder(View view) {
        orderMessage = getString(R.string.ice_cream_order_message);
        displayToast(orderMessage);
    }

    public void showFroyoOrder(View view) {
        orderMessage = getString(R.string.froyo_order_message);
        displayToast(orderMessage);
    }
}
