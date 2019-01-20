package za.co.simplesolutions.playapp;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.support.v4.app.ShareCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {

    private static final String LOG_TAG = MainActivity.class.getSimpleName();
    public static final String EXTRA_MESSAGE = "za.co.simplesolutions.extra.MESSAGE";
    private EditText websiteEditTxt;
    private EditText locationEditTxt;
    private EditText shareTextEditTxt;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        websiteEditTxt = findViewById(R.id.web_edittext);
        locationEditTxt = findViewById(R.id.location_edittext);
        shareTextEditTxt = findViewById(R.id.share_edittext);

        Log.i(MainActivity.class.getName(), "started...");
    }


    public void openWebsite(View view) {
        Log.d(LOG_TAG, "opening website");

        String url = websiteEditTxt.getText().toString();
        Uri webpage = Uri.parse(url);
        Intent intent = new Intent(Intent.ACTION_VIEW, webpage);
        if(intent.resolveActivity(getPackageManager()) != null) {
            startActivity(intent);
        }else {
            Log.d(LOG_TAG, "Cant handle this!");
        }
    }

    public void openLocation(View view) {
        Log.d(LOG_TAG, "opening location");
        String loc = locationEditTxt.getText().toString();
        Uri addressUri = Uri.parse("geo:0,0?q=" + loc);
        Intent intent = new Intent(Intent.ACTION_VIEW, addressUri);
        if(intent.resolveActivity(getPackageManager()) != null) {
            startActivity(intent);
        }
        else {
            Log.d(LOG_TAG, "Cant handle this intent!");
        }
    }

    public void shareText(View view) {
        Log.d(LOG_TAG, "sharing text");

        String text = shareTextEditTxt.getText().toString();
        String mimeTYpe = "text/plain";
        ShareCompat.IntentBuilder
                .from(this)
                .setType(mimeTYpe)
                .setChooserTitle(R.string.txt_chooser_title)
                .setText(text)
                .startChooser();
    }
}
